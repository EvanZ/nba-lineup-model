#!/usr/bin/env python3
"""Chronological shadow replay for in-season participation and minutes state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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

from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    UNKNOWN,
    InSeasonMinutesConfig,
    advance_minutes_state,
    forecast_team_minutes,
    initialize_minutes_state,
    publish_minutes_contribution,
    regulation_equivalent_minutes,
)
from nba_lineup_model.modeling.in_season_nail import read_game_contribution
from nba_lineup_model.season.slate_time import summary_slate_date

SEASON = "2026-27"
UPDATER_VERSION = "participation_conditional_shrinkage_v1"
ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NY": "NYK",
    "NO": "NOP",
    "SA": "SAS",
    "UT": "UTA",
    "UTAH": "UTA",
    "WSH": "WAS",
}
MEDICAL_TOKENS = (
    "INJURY",
    "ILLNESS",
    "CONCUSSION",
    "SORE",
    "SPRAIN",
    "STRAIN",
    "SURGERY",
    "RECOVERY",
)
INACTIVE_TOKENS = ("G LEAGUE", "ASSIGNMENT", "TWO-WAY", "SUSPENSION", "PERSONAL")


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
    espn_reason: object,
    injury_status: object,
    injury_reason: object,
) -> str:
    if minutes > 0.0:
        return PLAYED
    text = " ".join(
        str(value or "") for value in (espn_reason, injury_status, injury_reason)
    ).upper()
    if any(token in text for token in MEDICAL_TOKENS):
        return MEDICAL_ABSENCE
    if any(token in text for token in INACTIVE_TOKENS):
        return INACTIVE
    if "COACH" in text or (did_not_play and "DNP" in text):
        return COACH_DNP
    return UNKNOWN


def _pregame_rows(summary: dict[str, Any], team: str) -> list[dict[str, Any]]:
    side = "home" if str(summary["home_team"]) == team else "away"
    rows = summary.get("pregame", {}).get("projected_minutes", {}).get(side, [])
    return [row for row in rows if isinstance(row, dict)]


def _report_status(summary: dict[str, Any], team: str) -> str:
    rows = _pregame_rows(summary, team)
    if rows and all(str(row.get("status")) == "Report pending" for row in rows):
        return "not_yet_submitted"
    return "submitted"


def build_game_observations(
    summary: dict[str, Any],
    espn: dict[str, Any],
    *,
    priors: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize an immutable ESPN final into sufficient player observations."""

    name_to_id = {
        (str(row.team), _normalize_name(row.player_name)): int(row.player_id)
        for row in priors.itertuples(index=False)
    }
    injury = {
        int(row["player_id"]): row
        for team in (str(summary["home_team"]), str(summary["away_team"]))
        for row in _pregame_rows(summary, team)
        if row.get("player_id") is not None
    }
    rows: list[dict[str, object]] = []
    for group in espn.get("boxscore", {}).get("players", []):
        if not isinstance(group, dict):
            continue
        team_data = group.get("team", {})
        raw_team = str(team_data.get("abbreviation", "")) if isinstance(team_data, dict) else ""
        team = ESPN_TEAM_ALIASES.get(raw_team, raw_team)
        for statistics in group.get("statistics", []):
            if not isinstance(statistics, dict):
                continue
            labels = [str(label) for label in statistics.get("labels", [])]
            if "MIN" not in labels:
                continue
            minute_index = labels.index("MIN")
            team_rows: list[dict[str, object]] = []
            for athlete_row in statistics.get("athletes", []):
                if not isinstance(athlete_row, dict):
                    continue
                athlete = athlete_row.get("athlete", {})
                name = str(athlete.get("displayName", "")) if isinstance(athlete, dict) else ""
                player_id = name_to_id.get((team, _normalize_name(name)))
                if player_id is None:
                    continue
                stats = athlete_row.get("stats", [])
                minutes = (
                    _parse_minutes(stats[minute_index])
                    if isinstance(stats, list) and len(stats) > minute_index
                    else 0.0
                )
                injury_row = injury.get(player_id, {})
                classification = _classification(
                    minutes=minutes,
                    did_not_play=bool(athlete_row.get("didNotPlay")),
                    espn_reason=athlete_row.get("reason"),
                    injury_status=injury_row.get("status"),
                    injury_reason=injury_row.get("reason"),
                )
                weight = 0.5 if bool(athlete_row.get("ejected")) or 0.0 < minutes < 4.0 else 1.0
                team_rows.append(
                    {
                        "season": SEASON,
                        "game_id": str(summary["game_id"]),
                        "game_date": summary_slate_date(summary).isoformat(),
                        "team": team,
                        "player_id": player_id,
                        "player_name": name,
                        "actual_minutes": minutes,
                        "played": minutes > 0.0,
                        "outcome_classification": classification,
                        "conditional_observation_weight": weight if minutes > 0.0 else 0.0,
                        "espn_reason": athlete_row.get("reason"),
                        "ejected": bool(athlete_row.get("ejected")),
                        "injury_status": injury_row.get("status"),
                        "injury_reason": injury_row.get("reason"),
                    }
                )
            total = sum(float(row["actual_minutes"]) for row in team_rows)
            if total <= 0.0:
                raise ValueError(f"No actual player minutes for {team}/{summary['game_id']}")
            equivalents = regulation_equivalent_minutes(
                pd.Series([row["actual_minutes"] for row in team_rows]),
                team_total_minutes=total,
            )
            for row, equivalent in zip(team_rows, equivalents, strict=True):
                row["regulation_equivalent_minutes"] = float(equivalent)
                row["actual_team_minutes"] = total
            rows.extend(team_rows)
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError(f"No mapped player observations for {summary['game_id']}")
    return output.sort_values(["team", "player_id"], kind="stable").reset_index(drop=True)


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
    played = frame["played"].astype(bool)
    result: dict[str, object] = {}
    for label, prefix in (("frozen", "frozen"), ("updated", "updated")):
        conditional = frame.loc[played, f"{prefix}_conditional_minutes"].astype(float)
        actual_conditional = frame.loc[played, "regulation_equivalent_minutes"].astype(float)
        unconditional = frame[f"{prefix}_projected_minutes"].astype(float)
        actual = frame["regulation_equivalent_minutes"].astype(float)
        result[label] = {
            "conditional_minutes_n": int(played.sum()),
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    summaries = [
        _load_json(path) for path in sorted(Path(args.state_dir).glob("final/*.summary.json"))
    ]
    summaries = [
        row
        for row in summaries
        if row.get("final", {}).get("status", {}).get("completed")
        and str(row.get("game_id", "")).startswith("002")
        and summary_slate_date(row) <= date.fromisoformat(args.through_date)
    ]
    summaries.sort(
        key=lambda row: (
            summary_slate_date(row),
            str(row["pregame"]["start_time"]),
            row["game_id"],
        )
    )
    cache = _load_json(_minutes_cache_path(Path(args.release_root)))
    priors = pd.DataFrame(cache["minutes"]["players"])
    config = InSeasonMinutesConfig(
        updater_version=UPDATER_VERSION,
        participation_prior_games=float(args.participation_prior_games),
        conditional_prior_games=float(args.conditional_prior_games),
        rotation_size=int(cache["minutes"].get("initial_rotation_size", 15)),
    )
    state = initialize_minutes_state(priors, season=SEASON, config=config)
    ratings = dict(
        zip(
            priors["player_id"].astype(int),
            priors["projected_nail"].astype(float),
            strict=True,
        )
    )
    probability_scale = _probability_scale(summaries)
    rating_by_date = _rating_state_by_date(
        summaries,
        rating_cache_root=getattr(args, "rating_cache_dir", None),
        rating_coverage_path=getattr(args, "rating_coverage_path", None),
        prior_possessions=float(args.rating_prior_possessions),
    )
    grouped: dict[date, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        grouped[summary_slate_date(summary)].append(summary)
    observation_rows: list[pd.DataFrame] = []
    game_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, str]] = []
    created = 0
    cached = 0
    contribution_paths: list[Path] = []
    contribution_root = Path(args.cache_dir) / "contributions"
    for game_date in sorted(grouped):
        day_contributions: list[tuple[pd.DataFrame, str]] = []
        rating_adjustment = rating_by_date.get(game_date, {})
        for summary in grouped[game_date]:
            game_id = str(summary["game_id"])
            try:
                forecasts: dict[str, pd.DataFrame] = {}
                for team in (str(summary["away_team"]), str(summary["home_team"])):
                    forecasts[team] = forecast_team_minutes(
                        state,
                        team=team,
                        game_id=game_id,
                        forecast_cutoff=str(summary["pregame"]["frozen_at"]),
                        injury_rows=_pregame_rows(summary, team),
                        report_status=_report_status(summary, team),
                        config=config,
                    )
                espn_path = Path(args.state_dir) / "final" / f"{summary['espn_game_id']}.espn.json"
                espn = _load_json(espn_path)
                observations = build_game_observations(summary, espn, priors=priors)
                summary_path = (
                    Path(args.state_dir) / "final" / f"{summary['espn_game_id']}.summary.json"
                )
                source_sha = hashlib.sha256(
                    f"{_sha256(summary_path)}:{_sha256(espn_path)}".encode()
                ).hexdigest()
                metadata = {
                    "season": SEASON,
                    "game_id": game_id,
                    "game_date": game_date.isoformat(),
                    "source_game_date": str(summary["game_date"]),
                    "finalized_at": str(summary["archived_at"]),
                    "summary_sha256": _sha256(summary_path),
                    "espn_sha256": _sha256(espn_path),
                    "source_sha256": source_sha,
                    "updater_version": UPDATER_VERSION,
                }
                path, was_created, digest = publish_minutes_contribution(
                    observations, metadata, contribution_root
                )
                contribution_paths.append(path)
                created += int(was_created)
                cached += int(not was_created)
                day_contributions.append((observations, digest))
                actual = observations.set_index("player_id")
                for team in (str(summary["away_team"]), str(summary["home_team"])):
                    forecast = forecasts[team].set_index("player_id")
                    frozen_rows = {
                        int(row["player_id"]): row
                        for row in _pregame_rows(summary, team)
                    }
                    actual_team_ids = set(actual.loc[actual["team"].eq(team)].index)
                    player_ids = sorted(set(forecast.index) | actual_team_ids)
                    for player_id in player_ids:
                        actual_row = actual.loc[player_id] if player_id in actual.index else None
                        shadow_row = (
                            forecast.loc[player_id] if player_id in forecast.index else None
                        )
                        frozen = frozen_rows.get(player_id, {})
                        observation_rows.append(
                            pd.DataFrame(
                                [
                                    {
                                        "game_id": game_id,
                                        "game_date": str(game_date),
                                        "team": team,
                                        "player_id": player_id,
                                        "player_name": (
                                            str(shadow_row["player_name"])
                                            if shadow_row is not None
                                            else str(actual_row["player_name"])
                                        ),
                                        "outcome_classification": (
                                            str(actual_row["outcome_classification"])
                                            if actual_row is not None
                                            else UNKNOWN
                                        ),
                                        "played": (
                                            bool(actual_row["played"])
                                            if actual_row is not None
                                            else False
                                        ),
                                        "regulation_equivalent_minutes": (
                                            float(actual_row["regulation_equivalent_minutes"])
                                            if actual_row is not None
                                            else 0.0
                                        ),
                                        "frozen_participation_probability": float(
                                            frozen.get(
                                                "play_probability",
                                                shadow_row["preseason_participation_prior"],
                                            )
                                            or 0.0
                                        ),
                                        "updated_participation_probability": float(
                                            shadow_row["game_participation_probability"]
                                        ),
                                        "frozen_conditional_minutes": float(
                                            frozen.get(
                                                "conditional_minutes",
                                                shadow_row[
                                                    "preseason_conditional_minutes_prior"
                                                ],
                                            )
                                        ),
                                        "updated_conditional_minutes": float(
                                            shadow_row["current_conditional_minutes_estimate"]
                                        ),
                                        "frozen_projected_minutes": float(
                                            frozen.get("projected_minutes", 0.0)
                                        ),
                                        "updated_projected_minutes": float(
                                            shadow_row["final_normalized_minutes"]
                                        ),
                                    }
                                ]
                            )
                        )
                home = str(summary["home_team"])
                away = str(summary["away_team"])
                frozen_home = _frozen_map(summary, home)
                frozen_away = _frozen_map(summary, away)
                updated_home = _allocation_map(forecasts[home])
                updated_away = _allocation_map(forecasts[away])
                base_margin = float(summary["pregame"]["projection"]["home_margin"])
                frozen_strength = _signed_strength(summary, frozen_home, frozen_away, ratings)
                updated_strength = _signed_strength(summary, updated_home, updated_away, ratings)
                rating_only = base_margin + _signed_strength(
                    summary, frozen_home, frozen_away, rating_adjustment
                )
                minutes_only = base_margin + updated_strength - frozen_strength
                combined_ratings = {
                    player_id: rating + rating_adjustment.get(player_id, 0.0)
                    for player_id, rating in ratings.items()
                }
                combined = base_margin + _signed_strength(
                    summary, updated_home, updated_away, combined_ratings
                ) - frozen_strength
                margins = {
                    "frozen_frozen": base_margin,
                    "frozen_updated": minutes_only,
                    "updated_frozen": rating_only,
                    "updated_updated": combined,
                }
                row: dict[str, object] = {
                    "game_id": game_id,
                    "game_date": str(game_date),
                    "away_team": away,
                    "home_team": home,
                    "actual_margin": float(summary["final"]["actual_home_margin"]),
                }
                for label, margin in margins.items():
                    row[f"{label}_margin"] = margin
                    row[f"{label}_probability"] = float(expit(probability_scale * margin))
                    row[f"{label}_ats"] = _ats(summary, margin)
                game_rows.append(row)
                observation_rows[-1]["contribution_sha256"] = digest
            except Exception as error:
                exclusions.append({"game_id": game_id, "reason": str(error)})
        for observations, digest in day_contributions:
            state = advance_minutes_state(
                state, observations, config=config, contribution_sha256=digest
            )
    evaluation = (
        pd.concat(observation_rows, ignore_index=True)
        if observation_rows
        else pd.DataFrame()
    )
    games = pd.DataFrame(game_rows)
    player_metrics = _player_metrics(evaluation)
    four_cell = {
        label: _game_metrics(games, label)
        for label in (
            "frozen_frozen",
            "frozen_updated",
            "updated_frozen",
            "updated_updated",
        )
    }
    state["cutoff_time"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    state["prior_participation_weight"] = config.participation_prior_games
    state["prior_conditional_weight"] = config.conditional_prior_games
    result = {
        "schema_version": 1,
        "season": SEASON,
        "through_date": max(grouped).isoformat() if grouped else None,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "updater_version": UPDATER_VERSION,
        "config": {
            "participation_prior_games": config.participation_prior_games,
            "conditional_prior_games": config.conditional_prior_games,
            "rotation_size": config.rotation_size,
            "conditional_clip_delta": config.conditional_clip_delta,
        },
        "games": int(len(games)),
        "player_game_rows": int(len(evaluation)),
        "created_contributions": created,
        "cached_contributions": cached,
        "excluded_games": exclusions,
        "player_metrics": player_metrics,
        "four_cell_game_metrics": four_cell,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation.to_parquet(output / "player_game_evaluation.parquet", index=False)
    games.to_csv(output / "four_cell_game_evaluation.csv", index=False)
    state.to_parquet(output / "player_state.parquet", index=False)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    result["local_artifacts"] = {
        "summary": str(output / "summary.json"),
        "player_game_evaluation": str(output / "player_game_evaluation.parquet"),
        "four_cell_game_evaluation": str(output / "four_cell_game_evaluation.csv"),
        "player_state": str(output / "player_state.parquet"),
        "contribution_directories": [str(path) for path in dict.fromkeys(contribution_paths)],
    }
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--through-date", required=True)
    parser.add_argument("--participation-prior-games", type=float, required=True)
    parser.add_argument("--conditional-prior-games", type=float, required=True)
    parser.add_argument("--rating-cache-dir", type=Path)
    parser.add_argument("--rating-coverage-path", type=Path)
    parser.add_argument("--rating-prior-possessions", type=float, default=6000.0)
    return parser


if __name__ == "__main__":
    run(_parser().parse_args())
