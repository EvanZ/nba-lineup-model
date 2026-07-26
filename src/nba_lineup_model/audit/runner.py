from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from nba_lineup_model.audit.schema import AuditGameResult, AuditGameSpec, AuditManifest
from nba_lineup_model.build_game import GameReconstruction, reconstruct_game_payloads
from nba_lineup_model.ingest.nba_cdn import CachedResponse, NbaCdnClient, RawJsonCache
from nba_lineup_model.possessions import Possession

_POSSESSION_ESTIMATE_WARNING_TOLERANCE = 5.0
_DURATION_TOLERANCE_SECONDS = 0.01


class GamePayloadSource(Protocol):
    def fetch_play_by_play(
        self,
        game_id: str,
        *,
        use_cache: bool = True,
    ) -> CachedResponse: ...

    def fetch_boxscore(
        self,
        game_id: str,
        *,
        use_cache: bool = True,
    ) -> CachedResponse: ...


@dataclass(frozen=True)
class AuditRun:
    results: list[AuditGameResult]
    output_paths: dict[str, Path]


def audit_game_payloads(
    spec: AuditGameSpec,
    play_by_play_payload: Mapping[str, Any],
    boxscore_payload: Mapping[str, Any],
) -> AuditGameResult:
    """Run reconstruction and invariant checks for one manifest game."""

    reconstruction = reconstruct_game_payloads(
        play_by_play_payload,
        boxscore_payload,
    )
    return _audit_reconstruction(spec, reconstruction, boxscore_payload)


def run_audit_manifest(
    manifest: AuditManifest,
    *,
    client: GamePayloadSource,
    output_dir: Path | str = Path("data/audit"),
    use_cache: bool = True,
) -> AuditRun:
    """Fetch and audit every manifest game without writing full processed tables."""

    results: list[AuditGameResult] = []
    for spec in manifest.games:
        try:
            play_by_play = client.fetch_play_by_play(
                spec.game_id,
                use_cache=use_cache,
            )
            boxscore = client.fetch_boxscore(spec.game_id, use_cache=use_cache)
        except Exception as exc:
            results.append(_exception_result(spec, "fetch", exc))
            continue

        try:
            reconstruction = reconstruct_game_payloads(
                play_by_play.payload,
                boxscore.payload,
            )
        except Exception as exc:
            results.append(_exception_result(spec, "reconstruct", exc))
            continue

        try:
            results.append(
                _audit_reconstruction(
                    spec,
                    reconstruction,
                    boxscore.payload,
                )
            )
        except Exception as exc:
            results.append(_exception_result(spec, "audit", exc))

    root = Path(output_dir)
    output_paths = {
        "games": root / "games.parquet",
        "summary": root / "summary.parquet",
    }
    _write_parquet(audit_results_frame(results), output_paths["games"])
    _write_parquet(audit_summary_frame(results), output_paths["summary"])
    return AuditRun(results=results, output_paths=output_paths)


def audit_results_frame(results: Sequence[AuditGameResult]) -> pd.DataFrame:
    return pd.DataFrame(result.model_dump(mode="json") for result in results)


def audit_summary_frame(results: Sequence[AuditGameResult]) -> pd.DataFrame:
    rows = []
    grouped: dict[tuple[str, str, str], list[AuditGameResult]] = {}
    for result in results:
        key = (result.season, result.season_type, result.sample_group)
        grouped.setdefault(key, []).append(result)

    for (season, season_type, sample_group), group in sorted(grouped.items()):
        statuses = Counter(result.status for result in group)
        completed = [result for result in group if result.possession_count is not None]
        rows.append(
            {
                "season": season,
                "season_type": season_type,
                "sample_group": sample_group,
                "game_count": len(group),
                "pass_count": statuses["pass"],
                "warning_count": statuses["warning"],
                "fail_count": statuses["fail"],
                "error_count": statuses["error"],
                "overtime_count": sum(result.is_overtime is True for result in group),
                "pipeline_warning_count": sum(
                    result.lineup_warning_count
                    + result.possession_warning_count
                    + result.segment_warning_count
                    for result in group
                ),
                "pipeline_error_count": sum(
                    result.lineup_error_count
                    + result.possession_error_count
                    + result.segment_error_count
                    for result in group
                ),
                "mean_event_count": _mean(
                    [result.event_count for result in completed]
                ),
                "mean_possession_count": _mean(
                    [result.possession_count for result in completed]
                ),
                "mean_possession_segment_count": _mean(
                    [result.possession_segment_count for result in completed]
                ),
            }
        )
    return pd.DataFrame(rows)


def _audit_reconstruction(
    spec: AuditGameSpec,
    reconstruction: GameReconstruction,
    boxscore_payload: Mapping[str, Any],
) -> AuditGameResult:
    game = _boxscore_game(boxscore_payload)
    home_team = _mapping_field(game, "homeTeam")
    away_team = _mapping_field(game, "awayTeam")
    home_team_id = _int_field(home_team, "teamId")
    away_team_id = _int_field(away_team, "teamId")
    score_home = _int_field(home_team, "score")
    score_away = _int_field(away_team, "score")

    events = reconstruction.events
    possessions = reconstruction.possessions.possessions
    segments = reconstruction.possession_segments.segments
    final_event = events[-1]
    home_possession_count = sum(
        possession.offense_team_id == home_team_id for possession in possessions
    )
    away_possession_count = sum(
        possession.offense_team_id == away_team_id for possession in possessions
    )
    possession_count_difference = abs(home_possession_count - away_possession_count)
    segment_counts = Counter(segment.possession_index for segment in segments)
    source_change_count = sum(
        possession.terminal_reason == "source_change" for possession in possessions
    )

    score_matches_boxscore = (
        final_event.score_home == score_home and final_event.score_away == score_away
    )
    possession_score_conserved = (
        sum(possession.points_home for possession in possessions) == score_home
        and sum(possession.points_away for possession in possessions) == score_away
    )
    segment_score_conserved = _segment_scores_conserved(reconstruction)
    segment_duration_conserved = _segment_durations_conserved(reconstruction)
    balanced_possession_counts = _period_possession_counts_balanced(
        possessions,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
    )
    period_count = max(event.period for event in events)
    is_overtime = period_count > 4

    estimated_home = _estimated_possessions(home_team)
    estimated_away = _estimated_possessions(away_team)
    home_estimate_difference = home_possession_count - estimated_home
    away_estimate_difference = away_possession_count - estimated_away

    lineup_counts = _severity_counts(reconstruction.lineups.issues)
    possession_counts = _severity_counts(reconstruction.possessions.issues)
    segment_issue_counts = _severity_counts(reconstruction.possession_segments.issues)
    issue_codes = {
        *(f"lineup:{issue.code}" for issue in reconstruction.lineups.issues),
        *(f"possession:{issue.code}" for issue in reconstruction.possessions.issues),
        *(
            f"segment:{issue.code}"
            for issue in reconstruction.possession_segments.issues
        ),
    }
    failure_codes: set[str] = set()
    warning_codes: set[str] = set()

    if game.get("gameStatus") != 3:
        failure_codes.add("audit:boxscore_not_final")
    if not score_matches_boxscore:
        failure_codes.add("audit:event_score_mismatch")
    if not possession_score_conserved:
        failure_codes.add("audit:possession_score_conservation_failed")
    if not segment_score_conserved:
        failure_codes.add("audit:segment_score_conservation_failed")
    if not segment_duration_conserved:
        failure_codes.add("audit:segment_duration_conservation_failed")
    if not balanced_possession_counts:
        failure_codes.add("audit:unbalanced_period_possession_counts")
    if spec.expected_overtime is not None and spec.expected_overtime != is_overtime:
        failure_codes.add("audit:overtime_expectation_mismatch")
    if (
        abs(home_estimate_difference) > _POSSESSION_ESTIMATE_WARNING_TOLERANCE
        or abs(away_estimate_difference) > _POSSESSION_ESTIMATE_WARNING_TOLERANCE
    ):
        warning_codes.add("audit:possession_estimate_outlier")

    pipeline_error_count = (
        lineup_counts["error"]
        + possession_counts["error"]
        + segment_issue_counts["error"]
    )
    pipeline_warning_count = (
        lineup_counts["warning"]
        + possession_counts["warning"]
        + segment_issue_counts["warning"]
    )
    issue_codes.update(failure_codes)
    issue_codes.update(warning_codes)
    if pipeline_error_count or failure_codes:
        status = "fail"
    elif pipeline_warning_count or warning_codes:
        status = "warning"
    else:
        status = "pass"

    return AuditGameResult(
        game_id=spec.game_id,
        season=spec.season,
        season_type=spec.season_type,
        sample_group=spec.sample_group,
        status=status,
        game_time_utc=_optional_str(game.get("gameTimeUTC")),
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_tricode=_optional_str(home_team.get("teamTricode")),
        away_tricode=_optional_str(away_team.get("teamTricode")),
        score_home=score_home,
        score_away=score_away,
        period_count=period_count,
        is_overtime=is_overtime,
        event_count=len(events),
        lineup_stint_count=len(reconstruction.lineups.stints),
        possession_count=len(possessions),
        home_possession_count=home_possession_count,
        away_possession_count=away_possession_count,
        possession_count_difference=possession_count_difference,
        possession_segment_count=len(segments),
        multi_segment_possession_count=sum(count > 1 for count in segment_counts.values()),
        source_possession_override_count=sum(
            possession.source_possession_mismatch_count for possession in possessions
        ),
        source_change_terminal_count=source_change_count,
        opponent_technical_free_throw_possession_count=sum(
            "opponent_technical_free_throw" in possession.validation_flags
            for possession in possessions
        ),
        estimated_home_possessions=estimated_home,
        estimated_away_possessions=estimated_away,
        home_possession_estimate_difference=home_estimate_difference,
        away_possession_estimate_difference=away_estimate_difference,
        score_matches_boxscore=score_matches_boxscore,
        possession_score_conserved=possession_score_conserved,
        segment_score_conserved=segment_score_conserved,
        segment_duration_conserved=segment_duration_conserved,
        balanced_possession_counts=balanced_possession_counts,
        lineup_warning_count=lineup_counts["warning"],
        lineup_error_count=lineup_counts["error"],
        possession_warning_count=possession_counts["warning"],
        possession_error_count=possession_counts["error"],
        segment_warning_count=segment_issue_counts["warning"],
        segment_error_count=segment_issue_counts["error"],
        issue_codes=tuple(sorted(issue_codes)),
    )


def _segment_scores_conserved(reconstruction: GameReconstruction) -> bool:
    totals: dict[int, list[int]] = {}
    for segment in reconstruction.possession_segments.segments:
        points = totals.setdefault(segment.possession_index, [0, 0])
        points[0] += segment.points_home
        points[1] += segment.points_away
    return all(
        totals.get(possession.possession_index) == [
            possession.points_home,
            possession.points_away,
        ]
        for possession in reconstruction.possessions.possessions
    )


def _segment_durations_conserved(reconstruction: GameReconstruction) -> bool:
    totals: dict[int, float] = {}
    for segment in reconstruction.possession_segments.segments:
        totals[segment.possession_index] = (
            totals.get(segment.possession_index, 0.0) + segment.duration_seconds
        )
    return all(
        abs(totals.get(possession.possession_index, -1.0) - possession.duration_seconds)
        <= _DURATION_TOLERANCE_SECONDS
        for possession in reconstruction.possessions.possessions
    )


def _period_possession_counts_balanced(
    possessions: Sequence[Possession],
    *,
    home_team_id: int,
    away_team_id: int,
) -> bool:
    period_counts: dict[int, Counter[int]] = {}
    for possession in possessions:
        period_counts.setdefault(possession.period, Counter())[possession.offense_team_id] += 1
    return all(
        abs(counts[home_team_id] - counts[away_team_id]) <= 1
        for counts in period_counts.values()
    )


def _estimated_possessions(team: Mapping[str, Any]) -> float:
    statistics = _mapping_field(team, "statistics")
    field_goal_attempts = _number_field(statistics, "fieldGoalsAttempted")
    free_throw_attempts = _number_field(statistics, "freeThrowsAttempted")
    offensive_rebounds = _number_field(statistics, "reboundsOffensive")
    turnovers = _number_field(
        statistics,
        "turnoversTotal" if "turnoversTotal" in statistics else "turnovers",
    )
    return field_goal_attempts + 0.44 * free_throw_attempts - offensive_rebounds + turnovers


def _severity_counts(issues: Sequence[Any]) -> Counter[str]:
    return Counter(issue.severity for issue in issues)


def _exception_result(
    spec: AuditGameSpec,
    stage: str,
    exception: Exception,
) -> AuditGameResult:
    return AuditGameResult(
        game_id=spec.game_id,
        season=spec.season,
        season_type=spec.season_type,
        sample_group=spec.sample_group,
        status="error",
        issue_codes=(f"{stage}:{type(exception).__name__}",),
        error_stage=stage,
        error_type=type(exception).__name__,
        error_message=str(exception),
    )


def _boxscore_game(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping_field(payload, "game")


def _mapping_field(value: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise ValueError(f"Expected {field!r} to be an object")
    return result


def _int_field(value: Mapping[str, Any], field: str) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"Expected integer {field!r}, got {result!r}")
    return result


def _number_field(value: Mapping[str, Any], field: str) -> float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float):
        raise ValueError(f"Expected numeric {field!r}, got {result!r}")
    return float(result)


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _mean(values: Sequence[int | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit reconstruction invariants across a manifest of NBA games."
    )
    parser.add_argument("manifest", help="Versioned audit manifest JSON")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw JSON cache directory")
    parser.add_argument(
        "--output-dir",
        default="data/audit",
        help="Audit Parquet output directory",
    )
    parser.add_argument("--refresh", action="store_true", help="Ignore cached NBA responses")
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a non-zero exit status when any game has warnings",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = AuditManifest.read(args.manifest)
    client = NbaCdnClient(cache=RawJsonCache(Path(args.raw_dir)))
    run = run_audit_manifest(
        manifest,
        client=client,
        output_dir=Path(args.output_dir),
        use_cache=not args.refresh,
    )
    statuses = Counter(result.status for result in run.results)
    print(
        f"{len(run.results)} games: {statuses['pass']} passed, "
        f"{statuses['warning']} warnings, {statuses['fail']} failed, "
        f"{statuses['error']} errors"
    )
    for name, path in run.output_paths.items():
        print(f"{name}: {path}")

    should_fail = statuses["fail"] or statuses["error"]
    if args.fail_on_warnings:
        should_fail = should_fail or statuses["warning"]
    if should_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
