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
import urllib.error
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
from nba_lineup_model.season.slate_time import summary_slate_date

SEASON = "2026-27"
CORE_MODEL_VERSION = "forward_nail_rapm_v1212_residualized_lambda"
UPDATER_VERSION = "possession_residual_ridge_v1"


def _normalize_name(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", ascii_name.casefold()).strip()


def _name_aliases(value: object) -> tuple[str, ...]:
    normalized = _normalize_name(value)
    suffixless = re.sub(r"\s+(?:jr|sr|ii|iii|iv)$", "", normalized)
    return tuple(dict.fromkeys((normalized, suffixless)))


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "-", "none", "null"} else text


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










def _rapm_stints_from_nba_boxscore(
    *,
    boxscore: dict[str, Any],
    play_by_play: dict[str, Any],
    play_by_play_sha256: str,
    boxscore_sha256: str,
    game_date: str,
    game_time_utc: str,
) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Reconstruct RAPM stints using only canonical NBA liveData inputs."""

    game = boxscore.get("game")
    if not isinstance(game, dict):
        raise ValueError("NBA box score has no game payload")
    teams: dict[str, dict[str, Any]] = {}
    for side in ("homeTeam", "awayTeam"):
        team = game.get(side)
        if not isinstance(team, dict):
            raise ValueError(f"NBA box score has no {side}")
        tricode = str(team.get("teamTricode", "")).strip()
        if not tricode:
            raise ValueError(f"NBA box score {side} has no team tricode")
        teams[side] = team
    home = str(teams["homeTeam"]["teamTricode"])
    away = str(teams["awayTeam"]["teamTricode"])
    team_ids = {
        home: int(teams["homeTeam"]["teamId"]),
        away: int(teams["awayTeam"]["teamId"]),
    }
    return _rapm_stints_from_boxscore_context(
        starter_boxscore=boxscore,
        play_by_play=play_by_play,
        play_by_play_sha256=play_by_play_sha256,
        boxscore_sha256=boxscore_sha256,
        team_ids=team_ids,
        home=home,
        away=away,
        game_date=game_date,
        game_time_utc=game_time_utc,
        actual_margin=float(teams["homeTeam"]["score"])
        - float(teams["awayTeam"]["score"]),
    )


def _rapm_stints_from_boxscore_context(
    *,
    starter_boxscore: dict[str, Any],
    play_by_play: dict[str, Any],
    play_by_play_sha256: str,
    boxscore_sha256: str,
    team_ids: dict[str, int],
    home: str,
    away: str,
    game_date: str,
    game_time_utc: str,
    actual_margin: float,
) -> tuple[pd.DataFrame, dict[str, int], int]:
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
    for frame in (lineup_frame, segment_frame):
        frame["catalog_home_team_id"] = team_ids[home]
        frame["catalog_away_team_id"] = team_ids[away]
    lineup_frame = lineup_frame.assign(
        season=SEASON,
        season_type="regular",
        game_date=game_date,
        game_time_utc=game_time_utc,
        catalog_home_team_tricode=home,
        catalog_away_team_tricode=away,
        quality_status="accepted",
        quality_issue_codes_json="[]",
        source_build_run_id="current-season-updater-audit",
        processing_code_version=UPDATER_VERSION,
        play_by_play_sha256=play_by_play_sha256,
        boxscore_sha256=boxscore_sha256,
    )
    stints = rapm_stints_frame(lineup_frame, segment_frame)
    if not math.isclose(float(stints["home_margin"].sum()), actual_margin, abs_tol=1e-9):
        raise ValueError("Reconstructed stint margin does not match the immutable final")
    stints.attrs["reconstructed_possessions"] = len(
        reconstruction.possessions.possessions
    )
    stints.attrs["reconstruction_issues"] = [
        {
            "code": str(issue.code),
            "severity": str(issue.severity),
            "detail": str(issue.detail),
            "player_id": getattr(issue, "player_id", None),
        }
        for issue in issues
    ]
    return stints, team_ids, issue_count


def _b2b_flags(summaries: list[dict[str, Any]]) -> dict[str, tuple[bool, bool]]:
    previous_date: dict[str, date] = {}
    output: dict[str, tuple[bool, bool]] = {}
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[summary_slate_date(summary)].append(summary)
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
    cache: dict[tuple[object, ...], tuple[float, str | None]],
    court: str = "unit_home",
) -> tuple[np.ndarray, list[str]]:
    if court not in {"unit_home", "opponent_home", "neutral"}:
        raise ValueError(f"Unsupported court setting: {court}")
    offsets: list[float] = []
    fallback_reasons: set[str] = set()
    for row in stints.itertuples(index=False):
        home_ids = tuple(int(value) for value in row.home_player_ids)
        away_ids = tuple(int(value) for value in row.away_player_ids)
        key = (home_ids, away_ids, home_back_to_back, away_back_to_back, court)
        if key not in cache:
            try:
                response = _post_matchup(
                    matchup_url,
                    {
                        "unit_player_ids": home_ids,
                        "opponent_player_ids": away_ids,
                        "unit_season": SEASON,
                        "opponent_season": SEASON,
                        "environment": "unit",
                        "court": court,
                        "unit_back_to_back": home_back_to_back,
                        "opponent_back_to_back": away_back_to_back,
                        "include_response_curves": False,
                    },
                )
                cache[key] = (
                    float(response["predicted_net_rating"])
                    - float(response["additive_margin"]),
                    None,
                )
            except urllib.error.HTTPError as error:
                if error.code != 422:
                    raise
                detail = error.read().decode(errors="replace").strip()
                cache[key] = (
                    0.0,
                    f"matchup HTTP 422; neutral context fallback: {detail}",
                )
        offset, fallback_reason = cache[key]
        offsets.append(offset)
        if fallback_reason is not None:
            fallback_reasons.add(fallback_reason)
    return np.asarray(offsets, dtype=float), sorted(fallback_reasons)


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
    player_ids: tuple[int, ...],
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
            if contribution.player_ids != player_ids:
                continue
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        return contribution, manifest_path.parent
    return None, None


def build_player_game_trajectory(
    contributions_by_date: dict[date, list[Any]],
    *,
    player_ids: tuple[int, ...],
    preseason_nail: dict[int, float],
    prior_possessions: float,
    game_player_context: dict[tuple[str, int], dict[str, object]],
) -> pd.DataFrame:
    """Return the exact cumulative rating after each player team game.

    Updates remain slate-batched, matching the shadow updater's no-leakage
    contract. A player receives one row for every frozen-roster team game,
    including games missed through injury or DNP. Every row uses the cumulative
    solve after that game's full slate has been incorporated.
    """

    columns = [
        "season",
        "updater_version",
        "player_id",
        "game_number",
        "appearance_number",
        "game_id",
        "game_date",
        "game_time_utc",
        "team",
        "opponent",
        "played",
        "pregame_status",
        "pregame_reason",
        "preseason_rating",
        "in_season_delta",
        "updated_rating",
        "game_on_court_possessions",
        "cumulative_on_court_possessions",
        "effective_information",
        "evidence_weight",
    ]
    if not contributions_by_date:
        return pd.DataFrame(columns=columns)

    prior = np.asarray([preseason_nail[player_id] for player_id in player_ids], dtype=float)
    prior_precision = sparse.eye(len(player_ids), format="csr") * prior_possessions
    gram = sparse.csr_matrix((len(player_ids), len(player_ids)), dtype=float)
    score = np.zeros(len(player_ids), dtype=float)
    exposure = np.zeros(len(player_ids), dtype=float)
    game_numbers = np.zeros(len(player_ids), dtype=int)
    appearance_numbers = np.zeros(len(player_ids), dtype=int)
    player_to_index = {
        player_id: index for index, player_id in enumerate(player_ids)
    }
    rows: list[dict[str, object]] = []

    for game_date in sorted(contributions_by_date):
        day = sorted(
            contributions_by_date[game_date],
            key=lambda contribution: (contribution.game_time_utc, contribution.game_id),
        )
        if not day:
            continue
        for contribution in day:
            gram = gram + contribution.xtwx
            score += contribution.xtwy
            exposure += contribution.on_court_possessions
        adjustment = np.asarray(spsolve(gram + prior_precision, score), dtype=float)
        information = np.asarray(gram.diagonal(), dtype=float)
        evidence_weight = information / (information + prior_possessions)

        for contribution in day:
            game_context = {
                player_id: context
                for (game_id, player_id), context in game_player_context.items()
                if game_id == contribution.game_id and player_id in player_to_index
            }
            participants = {
                player_ids[index]
                for index in np.flatnonzero(contribution.on_court_possessions > 0)
            }
            missing_context = sorted(participants - set(game_context))
            if missing_context:
                raise ValueError(
                    "Missing team/opponent context for players "
                    f"{missing_context} in game {contribution.game_id}"
                )
            for player_id, context in sorted(game_context.items()):
                index = player_to_index[player_id]
                played = player_id in participants
                game_numbers[index] += 1
                if played:
                    appearance_numbers[index] += 1
                rows.append(
                    {
                        "season": SEASON,
                        "updater_version": UPDATER_VERSION,
                        "player_id": player_id,
                        "game_number": int(game_numbers[index]),
                        "appearance_number": (
                            int(appearance_numbers[index]) if played else None
                        ),
                        "game_id": contribution.game_id,
                        "game_date": game_date.isoformat(),
                        "game_time_utc": contribution.game_time_utc.isoformat(),
                        "team": context["team"],
                        "opponent": context["opponent"],
                        "played": played,
                        "pregame_status": context.get("pregame_status"),
                        "pregame_reason": context.get("pregame_reason"),
                        "preseason_rating": prior[index],
                        "in_season_delta": adjustment[index],
                        "updated_rating": prior[index] + adjustment[index],
                        "game_on_court_possessions": contribution.on_court_possessions[index],
                        "cumulative_on_court_possessions": exposure[index],
                        "effective_information": information[index],
                        "evidence_weight": evidence_weight[index],
                    }
                )
    return pd.DataFrame(rows, columns=columns)
