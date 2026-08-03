"""Import a shared Game Rotation CSV into the compatible local raw cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from nba_lineup_model.audit.game_rotation_probe import latest_failed_builds
from nba_lineup_model.ingest.nba_stats import (
    NbaStatsEndpoint,
    NbaStatsRawCache,
    StatsCachedResponse,
)
from nba_lineup_model.season.schema import CatalogGame, GameBuildRecord
from nba_lineup_model.season.storage import read_build_ledger, read_game_catalog

_INTERVAL_COLUMNS = ("game_id", "team_id", "player_id", "in_time_real", "out_time_real")
_PERIOD_START_TENTHS = {"period_2": 7200, "period_3": 14400, "period_4": 21600}


@dataclass(frozen=True)
class ExternalRotationImport:
    """One selected game and its externally supplied valid intervals."""

    game: CatalogGame
    intervals: pd.DataFrame
    discarded_nonpositive_intervals: int
    exact_period_starts: dict[str, bool]


def external_rotation_candidates(
    catalog: list[CatalogGame],
    failed_builds: list[GameBuildRecord],
    csv_path: Path | str,
) -> tuple[list[ExternalRotationImport], list[dict[str, object]]]:
    """Select failed regular-season games with exact five-player quarter starts."""

    catalog_by_id = {game.game_id: game for game in catalog}
    failed_games = {
        record.game_id
        for record in latest_failed_builds(failed_builds)
        if record.game_id in catalog_by_id
    }
    if not failed_games:
        return [], []

    intervals = _read_selected_intervals(Path(csv_path), failed_games)
    candidates: list[ExternalRotationImport] = []
    excluded: list[dict[str, object]] = []
    for game_id in sorted(failed_games):
        game = catalog_by_id[game_id]
        game_rows = intervals.loc[intervals["game_id"].eq(game_id)].copy()
        if game_rows.empty:
            excluded.append(_exclusion_row(game, "missing_csv_game"))
            continue
        valid = game_rows.loc[
            game_rows["out_time_real"].gt(game_rows["in_time_real"])
        ].copy()
        discarded = len(game_rows) - len(valid)
        expected_teams = {game.away_team_id, game.home_team_id}
        observed_teams = set(valid["team_id"].astype(int))
        if observed_teams != expected_teams:
            excluded.append(
                _exclusion_row(
                    game,
                    "team_mismatch",
                    discarded_nonpositive_intervals=discarded,
                    observed_team_ids=sorted(observed_teams),
                )
            )
            continue
        exact_period_starts = _exact_period_starts(valid, expected_teams)
        if not all(exact_period_starts.values()):
            excluded.append(
                _exclusion_row(
                    game,
                    "missing_exact_period_start",
                    discarded_nonpositive_intervals=discarded,
                    **exact_period_starts,
                )
            )
            continue
        candidates.append(
            ExternalRotationImport(
                game=game,
                intervals=valid,
                discarded_nonpositive_intervals=discarded,
                exact_period_starts=exact_period_starts,
            )
        )
    return candidates, excluded


def external_rotation_payload(imported: ExternalRotationImport) -> dict[str, object]:
    """Build the minimum official Game Rotation envelope consumed by the adapter."""

    headers = ["TEAM_ID", "PERSON_ID", "IN_TIME_REAL", "OUT_TIME_REAL"]
    result_sets: list[dict[str, object]] = []
    for name, team_id in (
        ("AwayTeam", imported.game.away_team_id),
        ("HomeTeam", imported.game.home_team_id),
    ):
        rows = imported.intervals.loc[
            imported.intervals["team_id"].eq(team_id),
            ["team_id", "player_id", "in_time_real", "out_time_real"],
        ].sort_values(["player_id", "in_time_real", "out_time_real"], kind="stable")
        if rows.empty:
            raise ValueError(f"External rotation has no intervals for {name}")
        result_sets.append(
            {
                "name": name,
                "headers": headers,
                "rowSet": rows.astype("int64").values.tolist(),
            }
        )
    return {
        "resource": "gamerotation",
        "parameters": {"GameID": imported.game.game_id, "LeagueID": "00"},
        "resultSets": result_sets,
    }


def import_external_game_rotations(
    csv_path: Path | str,
    *,
    catalog_path: Path | str = Path("data/catalog/games.parquet"),
    builds_path: Path | str = Path("data/manifests/builds.parquet"),
    raw_dir: Path | str = Path("data/raw"),
    output_dir: Path | str = Path("artifacts/reports/external_game_rotation_import"),
    limit: int | None = None,
    overwrite: bool = False,
    run_id: str | None = None,
) -> tuple[dict[str, object], Path]:
    """Cache structurally usable external rotations for latest regular failures."""

    source_path = Path(csv_path)
    if not source_path.is_file():
        raise ValueError(f"External Game Rotation CSV does not exist: {source_path}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    catalog = read_game_catalog(catalog_path).games
    failures = read_build_ledger(builds_path).records
    candidates, excluded = external_rotation_candidates(catalog, failures, source_path)
    if limit is not None:
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError("No latest failed regular-season games have usable external rotations")

    source_sha256 = _sha256_file(source_path)
    started_at = datetime.now(UTC)
    resolved_run_id = run_id or (
        f"external-game-rotation-import-{started_at:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    report_dir = Path(output_dir) / resolved_run_id
    report_dir.mkdir(parents=True, exist_ok=False)
    cache = NbaStatsRawCache(Path(raw_dir) / "stats")
    source_url = source_path.resolve().as_uri()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        game = candidate.game
        existing = cache.read(NbaStatsEndpoint.GAME_ROTATION, game.game_id)
        if existing is not None and not overwrite:
            rows.append(
                _import_row(
                    candidate,
                    "already_cached",
                    cache.path_for(NbaStatsEndpoint.GAME_ROTATION, game.game_id),
                )
            )
            continue
        payload = external_rotation_payload(candidate)
        raw_body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        cached_path = cache.write(
            StatsCachedResponse(
                endpoint=NbaStatsEndpoint.GAME_ROTATION,
                game_id=game.game_id,
                url=source_url,
                fetched_at=started_at,
                response_headers={
                    "x-nba-lineup-model-source": "external_game_rotation_csv",
                    "x-nba-lineup-model-source-filename": source_path.name,
                    "x-nba-lineup-model-source-sha256": source_sha256,
                },
                payload=payload,
                raw_body=raw_body,
            )
        )
        rows.append(_import_row(candidate, "imported", cached_path))

    pd.DataFrame(rows).to_parquet(report_dir / "games.parquet", index=False)
    pd.DataFrame(excluded).to_parquet(report_dir / "excluded_games.parquet", index=False)
    summary = {
        "run_id": resolved_run_id,
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "source_row_count": _csv_row_count(source_path),
        "selection_policy": "latest_regular_failures_with_exact_period_2_to_4_lineups",
        "selected_game_count": len(candidates),
        "imported_game_count": sum(row["cache_status"] == "imported" for row in rows),
        "already_cached_game_count": sum(row["cache_status"] == "already_cached" for row in rows),
        "excluded_game_count": len(excluded),
        "overwrite": overwrite,
        "created_at": started_at.isoformat(),
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, report_dir


def _read_selected_intervals(source_path: Path, game_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        source_path,
        usecols=list(_INTERVAL_COLUMNS),
        dtype={"game_id": "string"},
        chunksize=200_000,
    ):
        chunk["game_id"] = chunk["game_id"].str.zfill(10)
        selected = chunk.loc[chunk["game_id"].isin(game_ids)].copy()
        if selected.empty:
            continue
        for column in _INTERVAL_COLUMNS[1:]:
            selected[column] = pd.to_numeric(selected[column], errors="raise").astype("int64")
        if (selected["team_id"] <= 0).any() or (selected["player_id"] <= 0).any():
            raise ValueError("External Game Rotation contains nonpositive team or player IDs")
        if (selected[["in_time_real", "out_time_real"]] < 0).any().any():
            raise ValueError("External Game Rotation contains negative interval times")
        frames.append(selected)
    if not frames:
        return pd.DataFrame(columns=_INTERVAL_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _exact_period_starts(
    intervals: pd.DataFrame,
    expected_teams: set[int],
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for name, tenths in _PERIOD_START_TENTHS.items():
        active = intervals.loc[
            intervals["in_time_real"].le(tenths) & intervals["out_time_real"].gt(tenths)
        ].groupby("team_id")["player_id"].nunique()
        result[name] = all(active.get(team_id, 0) == 5 for team_id in expected_teams)
    return result


def _exclusion_row(game: CatalogGame, reason: str, **details: object) -> dict[str, object]:
    return {"game_id": game.game_id, "season": game.season, "reason": reason, **details}


def _import_row(
    candidate: ExternalRotationImport,
    cache_status: str,
    cache_path: Path,
) -> dict[str, object]:
    return {
        "game_id": candidate.game.game_id,
        "season": candidate.game.season,
        "cache_status": cache_status,
        "cache_path": str(cache_path),
        "interval_count": len(candidate.intervals),
        "discarded_nonpositive_intervals": candidate.discarded_nonpositive_intervals,
        **candidate.exact_period_starts,
    }


def _csv_row_count(path: Path) -> int:
    with path.open("rb") as source:
        return sum(1 for _line in source) - 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> None:
    """Run the external Game Rotation import."""

    parser = argparse.ArgumentParser(
        description="Import shared Game Rotation CSV evidence for latest regular-season failures."
    )
    parser.add_argument("csv_path")
    parser.add_argument("--catalog", default="data/catalog/games.parquet")
    parser.add_argument("--builds", default="data/manifests/builds.parquet")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="artifacts/reports/external_game_rotation_import")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    summary, report_dir = import_external_game_rotations(
        args.csv_path,
        catalog_path=args.catalog,
        builds_path=args.builds,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        overwrite=args.overwrite,
        run_id=args.run_id,
    )
    print(
        f"External Game Rotation import {summary['run_id']}: "
        f"selected={summary['selected_game_count']}, "
        f"imported={summary['imported_game_count']}, "
        f"cached={summary['already_cached_game_count']}, "
        f"excluded={summary['excluded_game_count']}; "
        f"report={report_dir}"
    )
