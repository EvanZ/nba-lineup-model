"""Forward-only evaluation for the isolated causal HCA candidate."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.daily_model_replay import (
    _component_path,
    _read_model_chain,
)

_OFFICIAL_SOURCE_PROVENANCE = {
    "basketball_facts": "NBA official liveData",
    "injuries": "NBA official injury report",
    "market": "The Odds API",
    "schedule_identity": "NBA official schedule",
}


def evaluate_forward_candidate(
    *,
    schedule_root: Path,
    candidate_model_root: Path,
    preseason_rankings_path: Path,
    hca_prior_mean: float,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Re-score immutable pregame inputs with causal D-1 ratings and HCA."""

    candidate_chain = _read_model_chain(candidate_model_root)
    if not candidate_chain:
        raise ValueError("Candidate model chain is empty")
    preseason = pd.read_parquet(
        preseason_rankings_path, columns=["player_id", "rapm"]
    )
    if preseason["player_id"].duplicated().any():
        raise ValueError("Preseason rankings contain duplicate player IDs")
    preseason_ratings = dict(
        zip(
            preseason["player_id"].astype(int),
            preseason["rapm"].astype(float),
            strict=True,
        )
    )
    states: list[tuple[date, dict[int, float], float, str]] = []
    for game_date, bundle, manifest in candidate_chain:
        ratings = pd.read_parquet(
            _component_path(bundle, manifest, "player_ratings"),
            columns=["player_id", "updated_nail"],
        )
        if ratings["player_id"].duplicated().any():
            raise ValueError(f"Candidate ratings duplicate players on {game_date}")
        hca = json.loads(
            _component_path(bundle, manifest, "home_court_state").read_text()
        )
        states.append(
            (
                game_date,
                dict(
                    zip(
                        ratings["player_id"].astype(int),
                        ratings["updated_nail"].astype(float),
                        strict=True,
                    )
                ),
                float(hca["home_court_estimate"]),
                str(manifest["bundle_sha256"]),
            )
        )

    summaries: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    final_root = schedule_root / "final"
    frozen_root = schedule_root / "frozen"
    for path in sorted(final_root.glob("*.summary.json")):
        summary = json.loads(path.read_text())
        game_date = date.fromisoformat(str(summary["game_date"]))
        if not start_date <= game_date <= end_date:
            continue
        game_id = str(summary["game_id"])
        if path.name != f"{game_id}.summary.json":
            raise ValueError(f"Final-summary identity mismatch for {path}")
        frozen_path = frozen_root / f"{game_id}.json"
        if not frozen_path.is_file():
            raise ValueError(f"Frozen pregame artifact is missing for {game_id}")
        frozen = json.loads(frozen_path.read_text())
        if summary.get("pregame") != frozen:
            raise ValueError(f"Final summary does not embed the frozen pregame for {game_id}")
        _validate_frozen_source(frozen, game_id=game_id)
        if not bool(summary.get("final", {}).get("status", {}).get("completed")):
            raise ValueError(f"Final summary is not completed for {game_id}")
        summaries.append((path, summary, frozen))
    if not summaries:
        raise ValueError("No completed frozen games fall within the evaluation window")
    expected_game_count = int(candidate_chain[-1][2]["identity"]["cumulative_game_count"])
    if len(summaries) != expected_game_count:
        raise ValueError(
            "Frozen/final evaluation coverage does not equal the candidate chain: "
            f"{len(summaries)} != {expected_game_count}"
        )

    beta = _probability_scale([frozen for _, _, frozen in summaries])
    rows: list[dict[str, Any]] = []
    for final_path, summary, frozen in summaries:
        game_date = date.fromisoformat(str(summary["game_date"]))
        prior_states = [row for row in states if row[0] < game_date]
        if prior_states:
            state_date, ratings, home_court, state_sha = prior_states[-1]
            state_as_of = state_date.isoformat()
        else:
            ratings = preseason_ratings
            home_court = float(hca_prior_mean)
            state_sha = "causal_hca_prior"
            state_as_of = (game_date - timedelta(days=1)).isoformat()
        projected = frozen.get("projected_minutes")
        if not isinstance(projected, dict):
            raise ValueError(f"Projected minutes are missing for {summary['game_id']}")
        home_strength = _project_team_strength(projected.get("home"), ratings)
        away_strength = _project_team_strength(projected.get("away"), ratings)
        ledger = frozen.get("projection", {}).get("ledger")
        if not isinstance(ledger, dict):
            raise ValueError(f"Frozen projection ledger is missing for {summary['game_id']}")
        b2b = ledger.get("back_to_back")
        if not isinstance(b2b, dict) or b2b.get("adjustment") is None:
            raise ValueError(f"Frozen B2B adjustment is missing for {summary['game_id']}")
        b2b_adjustment = float(b2b["adjustment"])
        candidate_margin = home_strength - away_strength + home_court + b2b_adjustment
        baseline_projection = frozen.get("projection")
        if not isinstance(baseline_projection, dict):
            raise ValueError(f"Frozen projection is missing for {summary['game_id']}")
        baseline_margin = float(baseline_projection["home_margin"])
        baseline_probability = float(baseline_projection["home_win_probability"])
        candidate_probability = float(1.0 / (1.0 + math.exp(-beta * candidate_margin)))
        actual_margin = float(summary["final"]["actual_home_margin"])
        market = frozen.get("market") if isinstance(frozen.get("market"), dict) else {}
        home_spread = (
            float(market["home_spread"])
            if market.get("home_spread") is not None
            else np.nan
        )
        rows.append(
            {
                "game_id": str(summary["game_id"]),
                "game_date": game_date.isoformat(),
                "away_team": str(summary["away_team"]),
                "home_team": str(summary["home_team"]),
                "rating_state_as_of": state_as_of,
                "rating_state_sha256": state_sha,
                "frozen_pregame_sha256": file_sha256(
                    schedule_root / "frozen" / f"{summary['game_id']}.json"
                ),
                "final_summary_sha256": file_sha256(final_path),
                "home_court_candidate": home_court,
                "back_to_back_adjustment": b2b_adjustment,
                "candidate_home_strength": home_strength,
                "candidate_away_strength": away_strength,
                "baseline_margin": baseline_margin,
                "candidate_margin": candidate_margin,
                "actual_margin": actual_margin,
                "baseline_probability": baseline_probability,
                "candidate_probability": candidate_probability,
                "market_probability": _market_home_probability(market),
                "home_spread": home_spread,
                "baseline_ats_result": _ats_result(
                    baseline_margin, actual_margin, home_spread
                ),
                "candidate_ats_result": _ats_result(
                    candidate_margin, actual_margin, home_spread
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["game_date", "game_id"], kind="stable"
    ).reset_index(drop=True)
    baseline = _metrics(frame, "baseline")
    candidate = _metrics(frame, "candidate")
    market_frame = frame.loc[frame["market_probability"].notna()].copy()
    market = _probability_metrics(market_frame, "market")
    return (
        {
            "status": "complete",
            "games": len(frame),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "probability_scale": beta,
            "baseline": baseline,
            "candidate": candidate,
            "market": market,
            "change_candidate_minus_baseline": {
                "margin_mae": candidate["margin_mae"] - baseline["margin_mae"],
                "margin_rmse": candidate["margin_rmse"] - baseline["margin_rmse"],
                "winner_accuracy": (
                    candidate["winner_accuracy"] - baseline["winner_accuracy"]
                ),
                "brier_score": candidate["brier_score"] - baseline["brier_score"],
                "log_loss": candidate["log_loss"] - baseline["log_loss"],
                "ats_wins": candidate["ats_wins"] - baseline["ats_wins"],
                "ats_losses": candidate["ats_losses"] - baseline["ats_losses"],
            },
            "frozen_input_contract": (
                "Same immutable projected minutes, injury assumptions, market line, "
                "and B2B adjustment; only D-1 player ratings and HCA are replaced."
            ),
            "source_provenance": dict(_OFFICIAL_SOURCE_PROVENANCE),
            "runtime_espn_dependencies": 0,
        },
        frame,
    )


def _validate_frozen_source(frozen: dict[str, Any], *, game_id: str) -> None:
    if str(frozen.get("game_id")) != game_id or not game_id.isdigit():
        raise ValueError(f"Frozen game identity is not an official NBA ID: {game_id}")
    provenance = frozen.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"Frozen source provenance is missing for {game_id}")
    for key, expected in _OFFICIAL_SOURCE_PROVENANCE.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"Frozen source {key} is not canonical for {game_id}: "
                f"{provenance.get(key)!r}"
            )
    if provenance.get("provider_fallbacks") not in (None, []):
        raise ValueError(f"Frozen provider fallback is present for {game_id}")
    if "espn" in canonical_json(frozen).decode().lower():
        raise ValueError(f"ESPN dependency found in frozen artifact {game_id}")


def _project_team_strength(
    projected_rows: object,
    ratings: dict[int, float],
) -> float:
    if not isinstance(projected_rows, list) or not projected_rows:
        raise ValueError("Projected team minutes must be a non-empty list")
    total_minutes = float(
        sum(float(row["projected_minutes"]) for row in projected_rows)
    )
    if not np.isclose(total_minutes, 240.0, atol=1e-6):
        raise ValueError(f"Projected team minutes do not sum to 240: {total_minutes}")
    missing = sorted(
        {
            int(row["player_id"])
            for row in projected_rows
            if int(row["player_id"]) not in ratings
        }
    )
    if missing:
        raise ValueError(f"Candidate ratings are missing players: {missing}")
    return float(
        sum(
            float(row["projected_minutes"])
            / 48.0
            * float(ratings[int(row["player_id"])])
            for row in projected_rows
        )
    )


def _probability_scale(frozen_games: list[dict[str, Any]]) -> float:
    values: list[float] = []
    for frozen in frozen_games:
        projection = frozen["projection"]
        margin = float(projection["home_margin"])
        probability = float(projection["home_win_probability"])
        if abs(margin) > 1e-8 and 0.0 < probability < 1.0:
            values.append(math.log(probability / (1.0 - probability)) / margin)
    if not values:
        raise ValueError("Could not recover the frozen win-probability scale")
    return float(np.median(values))


def _ats_result(predicted_margin: float, actual_margin: float, home_spread: float) -> str | None:
    if not np.isfinite(home_spread):
        return None
    pick_home = predicted_margin > -home_spread
    cover = actual_margin + home_spread
    selected_cover = cover if pick_home else -cover
    return "W" if selected_cover > 0 else "L" if selected_cover < 0 else "P"


def _metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    margin = frame[f"{prefix}_margin"].to_numpy(dtype=float)
    actual_margin = frame["actual_margin"].to_numpy(dtype=float)
    probability = frame[f"{prefix}_probability"].to_numpy(dtype=float)
    actual_win = (actual_margin > 0).astype(float)
    ats = frame[f"{prefix}_ats_result"].dropna()
    metrics = _probability_metrics(frame, prefix)
    return {
        "games": len(frame),
        "margin_mae": float(np.mean(np.abs(margin - actual_margin))),
        "margin_rmse": float(np.sqrt(np.mean(np.square(margin - actual_margin)))),
        "winner_accuracy": float(np.mean((margin > 0) == (actual_margin > 0))),
        "brier_score": metrics["brier_score"],
        "log_loss": metrics["log_loss"],
        "ats_wins": int(ats.eq("W").sum()),
        "ats_losses": int(ats.eq("L").sum()),
        "ats_pushes": int(ats.eq("P").sum()),
        "ats_games": int(len(ats)),
    }


def _probability_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, Any]:
    if frame.empty:
        return {"games": 0, "brier_score": None, "log_loss": None}
    probability = frame[f"{prefix}_probability"].to_numpy(dtype=float)
    actual_win = (frame["actual_margin"].to_numpy(dtype=float) > 0).astype(float)
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return {
        "games": len(frame),
        "brier_score": float(np.mean(np.square(probability - actual_win))),
        "log_loss": float(
            -np.mean(
                actual_win * np.log(clipped)
                + (1.0 - actual_win) * np.log(1.0 - clipped)
            )
        ),
    }


def _market_home_probability(market: dict[str, Any]) -> float:
    explicit = market.get("home_win_probability")
    if explicit is not None:
        value = float(explicit)
        if 0.0 < value < 1.0:
            return value
    home = _american_implied_probability(market.get("home_moneyline"))
    away = _american_implied_probability(market.get("away_moneyline"))
    if home is None or away is None or home + away <= 0:
        return np.nan
    return float(home / (home + away))


def _american_implied_probability(odds: object) -> float | None:
    if odds is None:
        return None
    value = float(odds)
    if value == 0:
        return None
    return -value / (-value + 100.0) if value < 0 else 100.0 / (value + 100.0)
