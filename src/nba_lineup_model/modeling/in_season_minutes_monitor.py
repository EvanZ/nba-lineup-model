#!/usr/bin/env python3
"""Chronological shadow replay for in-season participation and minutes state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import expit, logit

from nba_lineup_model.events.normalize import canonical_events
from nba_lineup_model.ingest.nba_cdn import NbaCdnEndpoint
from nba_lineup_model.lineups.reconstruct import lineup_stints_frame, reconstruct_lineups
from nba_lineup_model.modeling.game_script_minutes import (
    competitive_role_minutes,
)
from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    CONDITIONAL_OUTCOMES,
    G_LEAGUE_ASSIGNMENT,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    REGULATION_TEAM_MINUTES,
    UNKNOWN,
    UPDATER_VERSION,
    InSeasonMinutesConfig,
    advance_minutes_state,
    forecast_team_minutes,
    initialize_minutes_state,
    publish_minutes_contribution,
    regulation_equivalent_minutes,
)
from nba_lineup_model.modeling.in_season_minutes_report import (
    render_minutes_shadow_report,
)
from nba_lineup_model.modeling.in_season_nail import read_game_contribution
from nba_lineup_model.season.slate_time import summary_slate_date

SEASON = "2026-27"
DEFAULT_ROLE_CUTOFF_MINUTES = 44
DEFAULT_ROLE_LOOKBACK_GAMES = 3
DEFAULT_ROLE_PRIOR_GAMES = 0.2
G_LEAGUE_TOKENS = ("G LEAGUE", "G-LEAGUE", "GLEAGUE", "ASSIGNMENT", "TWO-WAY")
INACTIVE_TOKENS = ("SUSPENSION", "PERSONAL")
INJURY_REPORT_STATUSES = frozenset({"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"})


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _normalize_name(value: object) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    ascii_name = re.sub(r"\s+(?:Jr\.?|Sr\.?|II|III|IV)$", "", ascii_name, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", ascii_name.casefold())


def _parse_minutes(value: object) -> float:
    text = str(value or "0").strip()
    if ":" in text:
        minute, second = text.split(":", 1)
        return float(minute) + float(second) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minutes_cache_path(release_root: Path) -> Path:
    paths = sorted(release_root.glob("artifacts/web/win_projections/*/*.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one promoted minutes cache, found {len(paths)}")
    return paths[0]


def _classification(
    *,
    minutes: float,
    did_not_play: bool,
    boxscore_reason: object,
    injury_status: object,
    injury_reason: object = None,
) -> str:
    if minutes > 0.0:
        return PLAYED
    status = str(injury_status or "").strip().upper()
    text = " ".join(
        str(value).upper() for value in (boxscore_reason, injury_reason) if value
    )
    # The official report uses status=Out for G League/two-way assignments,
    # but this is a healthy team-deployment decision and therefore a
    # zero-minute available opportunity, not a medical absence.
    if any(token in text for token in G_LEAGUE_TOKENS):
        return G_LEAGUE_ASSIGNMENT
    if status in INJURY_REPORT_STATUSES:
        return MEDICAL_ABSENCE
    if any(token in text for token in INACTIVE_TOKENS):
        return INACTIVE
    if status == "AVAILABLE":
        return COACH_DNP
    if "COACH" in text or (did_not_play and "DNP" in text):
        return COACH_DNP
    return UNKNOWN


def _pregame_rows(summary: dict[str, Any], team: str) -> list[dict[str, Any]]:
    side = "home" if str(summary["home_team"]) == team else "away"
    rows = summary.get("pregame", {}).get("projected_minutes", {}).get(side, [])
    return [row for row in rows if isinstance(row, dict)]


def _pregame_roster_player_ids(summary: dict[str, Any], team: str) -> set[int]:
    """Return the immutable roster support known at the game's forecast cutoff."""

    player_ids = {
        int(row["player_id"])
        for row in _pregame_rows(summary, team)
        if row.get("player_id") is not None
    }
    if not player_ids:
        raise ValueError(f"Frozen pregame roster has no player IDs for {team}")
    return player_ids


def _report_status(summary: dict[str, Any], team: str) -> str:
    rows = _pregame_rows(summary, team)
    if rows and all(str(row.get("status")) == "Report pending" for row in rows):
        return "not_yet_submitted"
    return "submitted"


def _validate_non_replayed_source(summary: dict[str, Any]) -> None:
    """Reject a derived Schedule replay as an updater input boundary."""

    pregame = summary.get("pregame")
    pregame = pregame if isinstance(pregame, dict) else {}
    projection = pregame.get("projection")
    projection = projection if isinstance(projection, dict) else {}
    forecast_version = pregame.get("forecast_version")
    forecast_version = forecast_version if isinstance(forecast_version, dict) else {}
    if (
        str(summary.get("record_type")) == "strict_forward_replay_final_game_audit"
        or str(projection.get("state_kind")) == "in_season_strict_forward_replay"
        or str(forecast_version.get("mode")) == "strict_forward_replay"
    ):
        raise ValueError(
            f"Updater input {summary.get('game_id')} is a derived Schedule replay"
        )








def _load_cdn_play_by_play(
    game_id: str,
    *,
    coverage: dict[str, list[str]],
    lake_uri: str,
) -> tuple[dict[str, Any] | None, str | None]:
    digests = coverage.get(game_id, [])
    if not digests:
        return None, None
    digest = str(digests[-1])
    endpoint = NbaCdnEndpoint.PLAY_BY_PLAY.value
    uri = (
        f"{lake_uri.rstrip('/')}/raw/cdn/{endpoint}/season={SEASON}/"
        f"game_id={game_id}/{digest}.json"
    )
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--only-show-errors"],
        capture_output=True,
        check=True,
    )
    if hashlib.sha256(result.stdout).hexdigest() != digest:
        raise ValueError(f"CDN play-by-play checksum mismatch for {game_id}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError(f"CDN play-by-play is not an object for {game_id}")
    return payload, digest


def _probability_scale(summaries: list[dict[str, Any]]) -> float:
    values = []
    for summary in summaries:
        projection = summary["pregame"]["projection"]
        margin = float(projection["home_margin"])
        probability = float(projection["home_win_probability"])
        if abs(margin) > 1e-8 and 0.0 < probability < 1.0:
            values.append(float(logit(probability) / margin))
    if not values:
        raise ValueError("Could not recover frozen win-probability scale")
    return float(np.median(values))


def _latest_rating_contribution(
    root: Path,
    *,
    game_id: str,
    source_sha256: str,
) -> Any | None:
    game_root = root / "contributions" / f"season={SEASON}" / f"game_id={game_id}"
    matches: list[Any] = []
    for manifest in sorted(game_root.glob("contribution_sha256=*/manifest.json")):
        payload = _load_json(manifest)
        if str(payload.get("source_sha256")) == source_sha256:
            matches.append(read_game_contribution(manifest.parent))
    return max(matches, key=lambda item: len(item.player_ids)) if matches else None


def _rating_state_by_date(
    summaries: list[dict[str, Any]],
    *,
    rating_cache_root: Path | None,
    rating_coverage_path: Path | None,
    prior_possessions: float,
) -> dict[date, dict[int, float]]:
    dates = sorted({summary_slate_date(row) for row in summaries})
    if rating_cache_root is None or rating_coverage_path is None:
        return {game_date: {} for game_date in dates}
    coverage = _load_json(rating_coverage_path).get("games", {})
    contributions: dict[date, list[Any]] = defaultdict(list)
    player_ids: tuple[int, ...] | None = None
    for summary in summaries:
        game_id = str(summary["game_id"])
        versions = coverage.get(game_id, [])
        if not versions:
            continue
        contribution = _latest_rating_contribution(
            rating_cache_root, game_id=game_id, source_sha256=str(versions[-1])
        )
        if contribution is not None:
            player_ids = contribution.player_ids
            contributions[summary_slate_date(summary)].append(contribution)
    if player_ids is None:
        return {game_date: {} for game_date in dates}
    gram = sparse.csr_matrix((len(player_ids), len(player_ids)), dtype=float)
    score = np.zeros(len(player_ids), dtype=float)
    prior = sparse.eye(len(player_ids), format="csr") * float(prior_possessions)
    output: dict[date, dict[int, float]] = {}
    for game_date in dates:
        adjustment = (
            np.zeros(len(player_ids), dtype=float)
            if gram.nnz == 0
            else np.asarray(spsolve(gram + prior, score), dtype=float)
        )
        output[game_date] = dict(zip(player_ids, adjustment, strict=True))
        for contribution in contributions.get(game_date, []):
            gram = gram + contribution.xtwx
            score += contribution.xtwy
    return output


def _allocation_map(frame: pd.DataFrame) -> dict[int, float]:
    return dict(
        zip(
            frame["player_id"].astype(int),
            frame["final_normalized_minutes"].astype(float),
            strict=True,
        )
    )


def _validate_forecast_distribution(
    frame: pd.DataFrame, *, team: str, game_id: str
) -> None:
    """Fail closed on physically impossible pregame minute allocations."""

    minutes = pd.to_numeric(frame["final_normalized_minutes"], errors="raise")
    if not np.isfinite(minutes).all() or minutes.lt(-1e-9).any():
        raise ValueError(f"Non-finite or negative shadow minutes for {team}/{game_id}")
    total = float(minutes.sum())
    if not np.isclose(total, REGULATION_TEAM_MINUTES, atol=1e-6):
        raise ValueError(
            f"Shadow minutes do not conserve 240 for {team}/{game_id}: {total}"
        )
    maximum = float(minutes.max())
    if maximum > 48.0 + 1e-6:
        player = frame.loc[minutes.idxmax(), "player_name"]
        raise ValueError(
            f"Impossible shadow minutes for {player} ({team}/{game_id}): {maximum}"
        )


def _frozen_map(summary: dict[str, Any], team: str) -> dict[int, float]:
    return {
        int(row["player_id"]): float(row["projected_minutes"])
        for row in _pregame_rows(summary, team)
    }


def _signed_strength(
    summary: dict[str, Any],
    home: dict[int, float],
    away: dict[int, float],
    ratings: dict[int, float],
) -> float:
    del summary
    return float(
        sum(minutes / 48.0 * ratings.get(player_id, 0.0) for player_id, minutes in home.items())
        - sum(
            minutes / 48.0 * ratings.get(player_id, 0.0)
            for player_id, minutes in away.items()
        )
    )


def _projection_margin(
    summary: dict[str, Any],
    *,
    home_minutes: dict[int, float],
    away_minutes: dict[int, float],
    frozen_home: dict[int, float],
    frozen_away: dict[int, float],
    preseason_ratings: dict[int, float],
    forecast_ratings: dict[int, float],
) -> float:
    """Replace the frozen lineup strength with the requested ratings/minutes."""

    base_margin = float(summary["pregame"]["projection"]["home_margin"])
    frozen_strength = _signed_strength(summary, frozen_home, frozen_away, preseason_ratings)
    return base_margin + _signed_strength(
        summary, home_minutes, away_minutes, forecast_ratings
    ) - frozen_strength


def _ats(summary: dict[str, Any], margin: float) -> str | None:
    market = summary["pregame"].get("market") or {}
    spread = market.get("home_spread")
    if spread is None:
        return None
    market_margin = -float(spread)
    pick_home = margin > market_margin
    cover = float(summary["final"]["actual_home_margin"]) + float(spread)
    selected_cover = cover if pick_home else -cover
    return "W" if selected_cover > 0 else "L" if selected_cover < 0 else "P"


def _game_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, object]:
    margin = frame[f"{prefix}_margin"].astype(float).to_numpy()
    probability = frame[f"{prefix}_probability"].astype(float).to_numpy()
    actual_margin = frame["actual_margin"].astype(float).to_numpy()
    actual_win = (actual_margin > 0).astype(float)
    ats = frame[f"{prefix}_ats"].dropna()
    return {
        "games": int(len(frame)),
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
    }


def _player_metrics(frame: pd.DataFrame) -> dict[str, object]:
    available = frame["outcome_classification"].isin(CONDITIONAL_OUTCOMES)
    conditional_target = (
        "early_game_role_minutes"
        if "early_game_role_minutes" in frame
        else "regulation_equivalent_minutes"
    )
    result: dict[str, object] = {}
    for label, prefix in (("frozen", "frozen"), ("updated", "updated")):
        conditional = frame.loc[available, f"{prefix}_conditional_minutes"].astype(float)
        actual_conditional = frame.loc[available, conditional_target].astype(float)
        projected_column = f"{prefix}_projected_minutes"
        unconditional = frame[projected_column].astype(float)
        actual = frame["regulation_equivalent_minutes"].astype(float)
        result[label] = {
            "conditional_minutes_n": int(available.sum()),
            "conditional_minutes_mae": float(np.mean(np.abs(conditional - actual_conditional))),
            "conditional_minutes_rmse": float(
                np.sqrt(np.mean(np.square(conditional - actual_conditional)))
            ),
            "unconditional_minutes_mae": float(np.mean(np.abs(unconditional - actual))),
            "unconditional_minutes_rmse": float(
                np.sqrt(np.mean(np.square(unconditional - actual)))
            ),
        }
    return result
