#!/usr/bin/env python3
"""Replay and monitor the in-season NAIL updater in shadow mode.

The expensive game reconstruction is incremental: content-addressed sparse
contributions are reused on later runs. Predictions are always replayed in
date batches against immutable production forecasts, so the cumulative report
can be regenerated without allowing a game to learn from itself. Nothing in
this module changes the public projection path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import unicodedata
import urllib.request
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit

from nba_lineup_model.build_game import reconstruct_game_payloads
from nba_lineup_model.lineups import lineup_stints_frame
from nba_lineup_model.modeling.in_season_nail import (
    InSeasonNailConfig,
    build_game_contribution,
    publish_game_contribution,
    publish_player_state,
    read_game_contribution,
    solve_in_season_state,
)
from nba_lineup_model.modeling.stints import rapm_stints_frame
from nba_lineup_model.possessions import possession_segments_frame

SEASON = "2026-27"
CORE_MODEL_VERSION = "forward_nail_rapm_v1212_residualized_lambda"
UPDATER_VERSION = "possession_residual_ridge_v1"
ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS",
}


def _normalize_name(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def _name_aliases(value: object) -> tuple[str, ...]:
    normalized = _normalize_name(value)
    suffixless = re.sub(r"\s+(?:jr|sr|ii|iii|iv)$", "", normalized)
    return tuple(dict.fromkeys((normalized, suffixless)))


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _fetch_s3_json(uri: str) -> dict[str, Any]:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-"],
        check=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {uri}")
    return payload


def _post_matchup(url: str, request: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(request).encode()
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(http_request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Matchup endpoint returned a non-object response")
    return payload


def _ranking_paths(release_root: Path) -> tuple[Path, Path]:
    matches = sorted(
        release_root.glob(
            "artifacts/web/preseason_rankings/"
            "forward_nail_rapm_v1212_residualized_lambda/*/2026-27.parquet"
        )
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one promoted preseason ranking, found {len(matches)}")
    return matches[0], matches[0].with_suffix(".json")


def _player_name_lookup(rankings: pd.DataFrame) -> dict[str, int]:
    candidates: dict[str, set[int]] = defaultdict(set)
    for row in rankings.itertuples(index=False):
        for alias in _name_aliases(row.player_name):
            candidates[alias].add(int(row.player_id))
    return {
        alias: next(iter(player_ids))
        for alias, player_ids in candidates.items()
        if len(player_ids) == 1
    }


def _espn_team_rows(espn: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = espn.get("boxscore", {}).get("players", [])
    return {
        ESPN_TEAM_ALIASES.get(
            str(row.get("team", {}).get("abbreviation")),
            str(row.get("team", {}).get("abbreviation")),
        ): row
        for row in rows
        if isinstance(row, dict)
    }


def _native_starter_boxscore(
    *,
    summary: dict[str, Any],
    espn: dict[str, Any],
    play_by_play: dict[str, Any],
    name_to_player_id: dict[str, int],
) -> tuple[dict[str, Any], dict[str, int]]:
    actions = play_by_play.get("game", {}).get("actions", [])
    team_ids = {
        str(action.get("teamTricode")): int(action["teamId"])
        for action in actions
        if action.get("teamTricode") and action.get("teamId")
    }
    espn_teams = _espn_team_rows(espn)

    def team(side: str) -> dict[str, Any]:
        tricode = str(summary[f"{side}_team"])
        if tricode not in team_ids or tricode not in espn_teams:
            raise ValueError(f"Missing team metadata for {tricode}")
        starters: list[dict[str, object]] = []
        for statistic in espn_teams[tricode].get("statistics", []):
            for athlete_row in statistic.get("athletes", []):
                if not athlete_row.get("starter"):
                    continue
                player_name = athlete_row.get("athlete", {}).get("displayName")
                player_id = next(
                    (
                        name_to_player_id[alias]
                        for alias in _name_aliases(player_name)
                        if alias in name_to_player_id
                    ),
                    None,
                )
                if player_id is None:
                    raise ValueError(f"Could not map ESPN starter {player_name!r} for {tricode}")
                starters.append({"personId": player_id, "starter": "1"})
        if len(starters) != 5:
            raise ValueError(f"Expected five ESPN starters for {tricode}, found {len(starters)}")
        return {"teamId": team_ids[tricode], "players": starters}

    game_id = str(summary["game_id"])
    return (
        {
            "game": {
                "gameId": game_id,
                "homeTeam": team("home"),
                "awayTeam": team("away"),
            }
        },
        team_ids,
    )


def _rapm_stints(
    *,
    summary: dict[str, Any],
    espn: dict[str, Any],
    play_by_play: dict[str, Any],
    play_by_play_sha256: str,
    espn_sha256: str,
    name_to_player_id: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, int], int]:
    starter_boxscore, team_ids = _native_starter_boxscore(
        summary=summary,
        espn=espn,
        play_by_play=play_by_play,
        name_to_player_id=name_to_player_id,
    )
    reconstruction = reconstruct_game_payloads(play_by_play, starter_boxscore)
    issues = (
        *reconstruction.lineups.issues,
        *reconstruction.possessions.issues,
        *reconstruction.possession_segments.issues,
    )
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        codes = sorted({issue.code for issue in errors})
        raise ValueError(f"Reconstruction produced errors: {codes}")
    issue_count = len(issues)
    lineup_frame = lineup_stints_frame(reconstruction.lineups)
    segment_frame = possession_segments_frame(reconstruction.possession_segments)
    home = str(summary["home_team"])
    away = str(summary["away_team"])
    for frame in (lineup_frame, segment_frame):
        frame["catalog_home_team_id"] = team_ids[home]
        frame["catalog_away_team_id"] = team_ids[away]
    lineup_frame = lineup_frame.assign(
        season=SEASON,
        season_type="regular",
        game_date=str(summary["game_date"]),
        game_time_utc=str(summary["pregame"]["start_time"]),
        catalog_home_team_tricode=home,
        catalog_away_team_tricode=away,
        quality_status="accepted",
        quality_issue_codes_json="[]",
        source_build_run_id="current-season-updater-audit",
        processing_code_version=UPDATER_VERSION,
        play_by_play_sha256=play_by_play_sha256,
        boxscore_sha256=espn_sha256,
    )
    stints = rapm_stints_frame(lineup_frame, segment_frame)
    actual_margin = float(summary["final"]["actual_home_margin"])
    if not math.isclose(float(stints["home_margin"].sum()), actual_margin, abs_tol=1e-9):
        raise ValueError("Reconstructed stint margin does not match the immutable final")
    return stints, team_ids, issue_count


def _b2b_flags(summaries: list[dict[str, Any]]) -> dict[str, tuple[bool, bool]]:
    previous_date: dict[str, date] = {}
    output: dict[str, tuple[bool, bool]] = {}
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[date.fromisoformat(str(summary["game_date"]))].append(summary)
    for game_date in sorted(grouped):
        for summary in grouped[game_date]:
            home = str(summary["home_team"])
            away = str(summary["away_team"])
            output[str(summary["game_id"])] = (
                previous_date.get(home) is not None and (game_date - previous_date[home]).days == 1,
                previous_date.get(away) is not None and (game_date - previous_date[away]).days == 1,
            )
        for summary in grouped[game_date]:
            previous_date[str(summary["home_team"])] = game_date
            previous_date[str(summary["away_team"])] = game_date
    return output


def _context_offsets(
    stints: pd.DataFrame,
    *,
    matchup_url: str,
    home_back_to_back: bool,
    away_back_to_back: bool,
    cache: dict[tuple[object, ...], float],
) -> np.ndarray:
    offsets: list[float] = []
    for row in stints.itertuples(index=False):
        home_ids = tuple(int(value) for value in row.home_player_ids)
        away_ids = tuple(int(value) for value in row.away_player_ids)
        key = (home_ids, away_ids, home_back_to_back, away_back_to_back)
        if key not in cache:
            response = _post_matchup(
                matchup_url,
                {
                    "unit_player_ids": home_ids,
                    "opponent_player_ids": away_ids,
                    "unit_season": SEASON,
                    "opponent_season": SEASON,
                    "environment": "unit",
                    "court": "unit_home",
                    "unit_back_to_back": home_back_to_back,
                    "opponent_back_to_back": away_back_to_back,
                    "include_response_curves": False,
                },
            )
            cache[key] = float(response["predicted_net_rating"]) - float(
                response["additive_margin"]
            )
        offsets.append(cache[key])
    return np.asarray(offsets, dtype=float)


def _probability_scale(summaries: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for summary in summaries:
        projection = summary["pregame"]["projection"]
        margin = float(projection["home_margin"])
        probability = float(projection["home_win_probability"])
        if abs(margin) > 1e-8 and 0.0 < probability < 1.0:
            values.append(float(logit(probability) / margin))
    if not values:
        raise ValueError("Could not recover the frozen win-probability scale")
    return float(np.median(values))


def _minutes_adjustment(summary: dict[str, Any], adjustments: dict[int, float]) -> float:
    projected = summary["pregame"]["projected_minutes"]

    def side(rows: list[dict[str, Any]]) -> float:
        return float(
            sum(
                float(row["projected_minutes"]) / 48.0 * adjustments.get(int(row["player_id"]), 0.0)
                for row in rows
            )
        )

    return side(projected["home"]) - side(projected["away"])


def _ats_result(summary: dict[str, Any], predicted_margin: float) -> tuple[str | None, str | None]:
    market = summary["pregame"].get("market") or {}
    home_spread = market.get("home_spread")
    if home_spread is None:
        return None, None
    home_spread = float(home_spread)
    market_margin = -home_spread
    pick = str(summary["home_team"] if predicted_margin > market_margin else summary["away_team"])
    actual_with_home_spread = float(summary["final"]["actual_home_margin"]) + home_spread
    cover = actual_with_home_spread if pick == summary["home_team"] else -actual_with_home_spread
    return pick, "W" if cover > 0 else "L" if cover < 0 else "P"


def _metrics(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    margin = frame[f"{prefix}_margin"].to_numpy(dtype=float)
    probability = frame[f"{prefix}_probability"].to_numpy(dtype=float)
    actual_margin = frame["actual_margin"].to_numpy(dtype=float)
    actual_win = (actual_margin > 0).astype(float)
    ats = frame[f"{prefix}_ats_result"].dropna()
    return {
        "games": len(frame),
        "margin_mae": float(np.mean(np.abs(margin - actual_margin))),
        "margin_rmse": float(np.sqrt(np.mean(np.square(margin - actual_margin)))),
        "winner_accuracy": float(np.mean((margin > 0) == (actual_margin > 0))),
        "brier_score": float(np.mean(np.square(probability - actual_win))),
        "log_loss": float(
            -np.mean(
                actual_win * np.log(np.clip(probability, 1e-9, 1 - 1e-9))
                + (1 - actual_win) * np.log(np.clip(1 - probability, 1e-9, 1 - 1e-9))
            )
        ),
        "ats_wins": int(ats.eq("W").sum()),
        "ats_losses": int(ats.eq("L").sum()),
        "ats_pushes": int(ats.eq("P").sum()),
        "ats_games": int(len(ats)),
    }


def _cached_game_contribution(
    cache_root: Path,
    *,
    game_id: str,
    source_sha256: str,
) -> tuple[Any | None, Path | None]:
    game_root = cache_root / f"season={SEASON}" / f"game_id={game_id}"
    if not game_root.is_dir():
        return None, None
    for manifest_path in sorted(game_root.glob("contribution_sha256=*/manifest.json")):
        try:
            manifest = _load_json(manifest_path)
            if manifest.get("source_sha256") != source_sha256:
                continue
            contribution = read_game_contribution(manifest_path.parent)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        return contribution, manifest_path.parent
    return None, None


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_paths = sorted(args.state_dir.glob("final/*.summary.json"))
    summaries = [_load_json(path) for path in summary_paths]
    summaries = [
        summary
        for summary in summaries
        if summary.get("final", {}).get("status", {}).get("completed")
        and str(summary.get("game_id", "")).startswith("002")
    ]
    through_date = (
        date.fromisoformat(str(args.through_date)) if getattr(args, "through_date", "") else None
    )
    if through_date is not None:
        summaries = [
            summary
            for summary in summaries
            if date.fromisoformat(str(summary["game_date"])) <= through_date
        ]
    summaries.sort(
        key=lambda row: (
            str(row["game_date"]),
            str(row["pregame"]["start_time"]),
            str(row["game_id"]),
        )
    )
    rankings_path, _metadata_path = _ranking_paths(args.release_root)
    rankings = pd.read_parquet(rankings_path).sort_values("player_id", kind="stable")
    player_ids = tuple(rankings["player_id"].astype(int))
    preseason = dict(
        zip(rankings["player_id"].astype(int), rankings["rapm"].astype(float), strict=True)
    )
    name_lookup = _player_name_lookup(rankings)
    coverage = _load_json(args.coverage_path)
    b2b = _b2b_flags(summaries)
    probability_scale = _probability_scale(summaries)
    player_to_index = {player_id: index for index, player_id in enumerate(player_ids)}
    prior_precision = sparse.eye(len(player_ids), format="csr") * args.prior_possessions
    gram = sparse.csr_matrix((len(player_ids), len(player_ids)), dtype=float)
    score = np.zeros(len(player_ids), dtype=float)
    context_cache: dict[tuple[object, ...], float] = {}
    contributions_by_date: dict[date, list[Any]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    contribution_cache_root = Path(args.cache_dir) / "contributions"
    player_state_root = Path(args.cache_dir) / "player_states"
    cached_contribution_count = 0
    created_contribution_paths: list[Path] = []
    contribution_paths: list[Path] = []

    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[date.fromisoformat(str(summary["game_date"]))].append(summary)

    for game_date in sorted(grouped):
        adjustment_vector = (
            np.zeros(len(player_ids), dtype=float)
            if gram.nnz == 0
            else np.asarray(spsolve(gram + prior_precision, score), dtype=float)
        )
        adjustment_map = dict(zip(player_ids, adjustment_vector, strict=True))
        day_contributions: list[Any] = []
        for summary in grouped[game_date]:
            game_id = str(summary["game_id"])
            current_margin = float(summary["pregame"]["projection"]["home_margin"])
            current_probability = float(summary["pregame"]["projection"]["home_win_probability"])
            delta = _minutes_adjustment(summary, adjustment_map)
            updated_margin = current_margin + delta
            updated_probability = float(expit(probability_scale * updated_margin))
            current_pick, current_ats = _ats_result(summary, current_margin)
            updated_pick, updated_ats = _ats_result(summary, updated_margin)
            rows.append(
                {
                    "game_date": str(game_date),
                    "game_id": game_id,
                    "away_team": summary["away_team"],
                    "home_team": summary["home_team"],
                    "actual_margin": float(summary["final"]["actual_home_margin"]),
                    "current_margin": current_margin,
                    "updated_margin": updated_margin,
                    "margin_update": delta,
                    "current_probability": current_probability,
                    "updated_probability": updated_probability,
                    "current_ats_pick": current_pick,
                    "updated_ats_pick": updated_pick,
                    "current_ats_result": current_ats,
                    "updated_ats_result": updated_ats,
                    "prior_games": int(
                        sum(len(values) for values in contributions_by_date.values())
                    ),
                }
            )

            try:
                espn_path = args.state_dir / "final" / f"{summary['espn_game_id']}.espn.json"
                espn_bytes = espn_path.read_bytes()
                espn = json.loads(espn_bytes)
                source_versions = coverage.get("games", {}).get(game_id, [])
                if not source_versions:
                    raise ValueError("No CDN play-by-play checksum in coverage manifest")
                source_sha = str(source_versions[-1])
                source_uri = (
                    f"{args.cdn_s3_prefix}/season={SEASON}/game_id={game_id}/{source_sha}.json"
                )
                cached, cached_path = _cached_game_contribution(
                    contribution_cache_root,
                    game_id=game_id,
                    source_sha256=source_sha,
                )
                if cached is not None:
                    day_contributions.append(cached)
                    if cached_path is None:
                        raise RuntimeError("Cached contribution is missing its directory")
                    contribution_paths.append(cached_path)
                    cached_contribution_count += 1
                    continue
                play_by_play = _fetch_s3_json(source_uri)
                stints, _team_ids, _issue_count = _rapm_stints(
                    summary=summary,
                    espn=espn,
                    play_by_play=play_by_play,
                    play_by_play_sha256=source_sha,
                    espn_sha256=_sha256_bytes(espn_bytes),
                    name_to_player_id=name_lookup,
                )
                participants = {
                    int(player_id)
                    for column in ("home_player_ids", "away_player_ids")
                    for lineup in stints[column]
                    for player_id in lineup
                }
                missing = sorted(participants - set(player_to_index))
                if missing:
                    raise ValueError(f"Players missing from frozen preseason prior: {missing}")
                home_b2b, away_b2b = b2b[game_id]
                fixed_context = _context_offsets(
                    stints,
                    matchup_url=args.matchup_url,
                    home_back_to_back=home_b2b,
                    away_back_to_back=away_b2b,
                    cache=context_cache,
                )
                start_time = _parse_time(summary["pregame"]["start_time"])
                finalized_at = _parse_time(summary["archived_at"])
                if finalized_at < start_time:
                    finalized_at = start_time
                contribution = build_game_contribution(
                    stints,
                    season=SEASON,
                    game_id=game_id,
                    game_time_utc=start_time,
                    finalized_at=finalized_at,
                    source_sha256=source_sha,
                    source_uri=source_uri,
                    core_model_version=CORE_MODEL_VERSION,
                    context_model_version=CORE_MODEL_VERSION + ":2025-26",
                    player_ids=player_ids,
                    preseason_nail=preseason,
                    fixed_context_offset=fixed_context,
                )
                contribution_path, created = publish_game_contribution(
                    contribution,
                    contribution_cache_root,
                )
                day_contributions.append(contribution)
                contribution_paths.append(contribution_path)
                if created:
                    created_contribution_paths.append(contribution_path)
            except Exception as error:  # Keep the audit report explicit about exclusions.
                exclusions.append({"game_id": game_id, "reason": str(error)})

        for contribution in day_contributions:
            gram = gram + contribution.xtwx
            score += contribution.xtwy
        contributions_by_date[game_date].extend(day_contributions)

    predictions = pd.DataFrame(rows)
    metrics = {
        "current_model": _metrics(predictions, "current"),
        "in_season_updater": _metrics(predictions, "updated"),
    }
    all_contributions = [
        contribution
        for game_date in sorted(contributions_by_date)
        for contribution in contributions_by_date[game_date]
    ]
    generated_at = datetime.now(UTC)
    state = solve_in_season_state(
        season=SEASON,
        cutoff_time=generated_at + timedelta(microseconds=1),
        player_ids=player_ids,
        preseason_nail=preseason,
        contributions=all_contributions,
        config=InSeasonNailConfig(
            core_model_version=CORE_MODEL_VERSION,
            updater_version=UPDATER_VERSION,
            prior_equivalent_possessions=float(args.prior_possessions),
        ),
        team_by_player=dict(
            zip(rankings["player_id"].astype(int), rankings["team"].astype(str), strict=True)
        ),
        cutoff_game_id=(str(summaries[-1]["game_id"]) if summaries else None),
    )
    player_state_path, player_state_created = publish_player_state(state, player_state_root)
    result = {
        "season": SEASON,
        "through_date": max(grouped).isoformat() if grouped else None,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "method": {
            "prior_equivalent_possessions": args.prior_possessions,
            "date_batched": True,
            "frozen_pregame_minutes": True,
            "actual_lineup_stints_for_updates": True,
            "win_probability_scale": probability_scale,
        },
        "game_count": len(predictions),
        "contribution_count": sum(len(values) for values in contributions_by_date.values()),
        "cached_contribution_count": cached_contribution_count,
        "created_contribution_count": len(created_contribution_paths),
        "excluded_contributions": exclusions,
        "shadow_player_state": {
            "player_count": len(state),
            "state_sha256": player_state_path.name.removeprefix("state_sha256="),
            "cutoff_time": generated_at.isoformat().replace("+00:00", "Z"),
            "created": player_state_created,
        },
        "metrics": metrics,
        "comparison": {
            "margin_mae_change": (
                metrics["in_season_updater"]["margin_mae"] - metrics["current_model"]["margin_mae"]
            ),
            "margin_rmse_change": (
                metrics["in_season_updater"]["margin_rmse"]
                - metrics["current_model"]["margin_rmse"]
            ),
            "brier_change": (
                metrics["in_season_updater"]["brier_score"]
                - metrics["current_model"]["brier_score"]
            ),
            "mean_absolute_projection_change": float(predictions["margin_update"].abs().mean()),
            "max_absolute_projection_change": float(predictions["margin_update"].abs().max()),
            "winner_pick_changes": int(
                ((predictions["current_margin"] > 0) != (predictions["updated_margin"] > 0)).sum()
            ),
            "ats_pick_changes": int(
                predictions["current_ats_pick"].ne(predictions["updated_ats_pick"]).sum()
            ),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "game_predictions.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    result["local_artifacts"] = {
        "summary": str(args.output_dir / "summary.json"),
        "game_predictions": str(args.output_dir / "game_predictions.csv"),
        "player_state": str(player_state_path / "player_state.parquet"),
        "contribution_directories": [
            str(path) for path in dict.fromkeys(contribution_paths)
        ],
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("/var/lib/nba-gestalt/schedule"))
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path("/var/lib/nba-gestalt/cdn/coverage.json"),
    )
    parser.add_argument("--release-root", type=Path, default=Path("/opt/nba-gestalt/current"))
    parser.add_argument(
        "--cdn-s3-prefix",
        default="s3://nba-gestalt-data/lake/raw/cdn/playbyplay",
    )
    parser.add_argument("--matchup-url", default="http://127.0.0.1:8010/api/matchups")
    parser.add_argument("--prior-possessions", type=float, default=6000.0)
    parser.add_argument("--through-date", default="")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/var/lib/nba-gestalt/in-season-nail"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/current-season-nail-audit"))
    return parser


def main() -> None:
    audit = run(_parser().parse_args())
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
