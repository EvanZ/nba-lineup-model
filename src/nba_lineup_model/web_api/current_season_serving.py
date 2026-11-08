"""Load the promoted current-season state for public API responses.

The completed-season model artifact remains immutable.  This adapter overlays
only the promoted scalar rating and conditional-minutes state on the frozen
preseason catalog, while retaining the preseason values for auditability.
"""

from __future__ import annotations

from nba_lineup_model.storage import website_data as data_io

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.season.current_serving import (
    RATING_TRAJECTORY_FILENAME,
    SERVING_STATE_FILENAME,
    resolve_latest_bundle,
    resolve_pointer,
    verify_serving_bundle,
)

DEFAULT_CURRENT_SEASON_SERVING_ROOT = Path("/var/lib/nba-gestalt/current-season-serving")


@dataclass(frozen=True)
class CurrentSeasonServingCatalog:
    """One verified current state plus its promoted daily trajectory."""

    season: str
    rankings: pd.DataFrame
    metadata: dict[str, Any]
    trajectories: dict[int, list[dict[str, Any]]]
    bundle_path: Path


def load_current_season_serving_catalog(
    preseason_rankings: pd.DataFrame,
    *,
    root: Path | str | None = None,
    bundle_path: Path | str | None = None,
) -> CurrentSeasonServingCatalog | None:
    """Load and reconcile the promoted state with its exact preseason catalog.

    A missing promotion is intentionally non-fatal so a release can be started
    before the first in-season state is promoted.  A present but invalid state
    fails closed instead of silently returning stale preseason ratings.
    """

    configured_bundle = bundle_path or os.environ.get("GESTALT_CURRENT_SEASON_SERVING_BUNDLE")
    root_path = data_io.resolve(Path(
        root
        or os.environ.get(
            "GESTALT_CURRENT_SEASON_SERVING_ROOT",
            DEFAULT_CURRENT_SEASON_SERVING_ROOT,
        )
    ))
    if configured_bundle:
        latest_bundle = data_io.resolve(Path(configured_bundle))
        if not data_io.is_dir(latest_bundle):
            raise FileNotFoundError(f"Configured current-season bundle is missing: {latest_bundle}")
    else:
        if not data_io.is_file(root_path / "latest.json"):
            return None
        latest_bundle = resolve_latest_bundle(root_path)

    latest_manifest = verify_serving_bundle(latest_bundle)
    identity = latest_manifest["identity"]
    season = str(identity["season"])
    latest_state = data_io.parquet(latest_bundle / SERVING_STATE_FILENAME)
    rankings = _overlay_current_state(preseason_rankings, latest_state, identity)
    history_paths = _promoted_bundle_history(
        root_path,
        latest_bundle=latest_bundle,
        latest_manifest=latest_manifest,
        explicit_bundle=bool(configured_bundle),
    )
    trajectory_path = latest_bundle / RATING_TRAJECTORY_FILENAME
    trajectories = (
        _player_game_rating_trajectories(
            rankings,
            data_io.parquet(trajectory_path),
            current_bundle_sha=str(latest_manifest["bundle_sha256"]),
            evidence_through_date=str(identity["evidence_through_date"]),
            as_of_cutoff_time=str(identity["as_of_cutoff_time"]),
        )
        if data_io.is_file(trajectory_path)
        else _rating_trajectories(
            rankings,
            history_paths,
            current_bundle_sha=str(latest_manifest["bundle_sha256"]),
        )
    )
    coverage = identity["coverage"]
    sources = identity["sources"]
    metadata = {
        "kind": "in_season_updated",
        "season": season,
        "bundle_sha256": str(latest_manifest["bundle_sha256"]),
        "evidence_through_date": str(identity["evidence_through_date"]),
        "as_of_cutoff_time": str(identity["as_of_cutoff_time"]),
        "preseason_run_id": str(identity["preseason_run_id"]),
        "game_count": int(coverage["game_count"]),
        "player_count": int(coverage["serving_player_count"]),
        "source_game_ids": [str(game_id) for game_id in coverage["source_game_ids"]],
        "rating_updater_version": str(identity["sources"]["rating_state"]["updater_version"]),
        "minutes_updater_version": str(identity["sources"]["minutes_state"]["updater_version"]),
        "artifact_sha256": {
            "rating_state": str(sources["rating_state"]["sha256"]),
            "minutes_state": str(sources["minutes_state"]["sha256"]),
        },
        "artifact_sources": {
            "rating_state": str(sources["rating_state"]["uri"]),
            "minutes_state": str(sources["minutes_state"]["uri"]),
        },
        "data_quality": identity.get(
            "data_quality",
            {"status": "unknown", "issue_count": None},
        ),
        "schedule_controls": identity.get("schedule_controls"),
        "additive_profiles": "frozen_preseason",
        "offense_defense_split": "unavailable",
    }
    return CurrentSeasonServingCatalog(
        season=season,
        rankings=rankings,
        metadata=metadata,
        trajectories=trajectories,
        bundle_path=latest_bundle,
    )


def _overlay_current_state(
    preseason_rankings: pd.DataFrame,
    state: pd.DataFrame,
    identity: dict[str, Any],
) -> pd.DataFrame:
    required_preseason = {"season", "player_id", "player_name", "team", "rapm"}
    missing = sorted(required_preseason - set(preseason_rankings.columns))
    if missing:
        raise ValueError("Preseason rankings are missing: " + ", ".join(missing))
    if preseason_rankings["player_id"].duplicated().any():
        raise ValueError("Preseason rankings contain duplicate player IDs")
    season = str(identity["season"])
    baseline = preseason_rankings.loc[
        preseason_rankings["season"].astype(str).eq(season)
    ].copy()
    if baseline.empty:
        raise ValueError(f"Preseason rankings are unavailable for {season}")
    state_ids = set(state["player_id"].astype(int))
    baseline_ids = set(baseline["player_id"].astype(int))
    missing_baselines = sorted(state_ids - baseline_ids)
    if missing_baselines:
        raise ValueError(
            "Current-season players are missing frozen preseason ratings: "
            + ", ".join(map(str, missing_baselines))
        )

    state_columns = [
        "player_id",
        "player_name",
        "team",
        "frozen_preseason_nail",
        "current_season_residual",
        "updated_nail",
        "raw_on_court_possessions",
        "effective_information",
        "prior_weight",
        "evidence_weight",
        "posterior_standard_error",
        "current_conditional_minutes_estimate",
        "preseason_conditional_minutes_prior",
        "conditional_observations",
        "effective_conditional_weight",
        "current_participation_estimate",
        "participation_observations",
    ]
    current = state.loc[:, state_columns].rename(
        columns={
            "player_name": "current_player_name",
            "team": "current_team",
            "frozen_preseason_nail": "preseason_nail",
            "current_season_residual": "in_season_delta",
            "raw_on_court_possessions": "current_season_possessions",
            "effective_information": "rating_effective_information",
            "prior_weight": "rating_prior_weight",
            "evidence_weight": "rating_evidence_weight",
            "posterior_standard_error": "rating_posterior_standard_error",
            "current_conditional_minutes_estimate": "current_expected_minutes_if_available",
            "preseason_conditional_minutes_prior": "preseason_expected_minutes_if_available",
            "conditional_observations": "minutes_role_observations",
            "effective_conditional_weight": "minutes_effective_observation_weight",
            "current_participation_estimate": "current_participation_estimate",
            "participation_observations": "participation_observations",
        }
    )
    rankings = baseline.loc[baseline["player_id"].astype(int).isin(state_ids)].merge(
        current,
        on="player_id",
        how="inner",
        validate="one_to_one",
    )
    if len(rankings) != len(state):
        raise ValueError("Current-season ranking merge did not preserve every serving player")
    if not np.allclose(
        rankings["rapm"].to_numpy(dtype=float),
        rankings["preseason_nail"].to_numpy(dtype=float),
        atol=1e-10,
    ):
        raise ValueError("Serving-state preseason ratings do not match the promoted model release")
    rankings["player_name"] = rankings.pop("current_player_name")
    rankings["team"] = rankings.pop("current_team")
    rankings["rapm"] = rankings.pop("updated_nail")
    rankings["possessions"] = rankings["current_season_possessions"]
    rankings["offense_rating"] = np.nan
    rankings["defense_rating"] = np.nan
    rankings["rating_state"] = "in_season_updated"
    rankings["additive_profile_state"] = "frozen_preseason"
    rankings["offense_defense_state"] = "unavailable"
    rankings["rating_evidence_through_date"] = str(identity["evidence_through_date"])
    rankings["rating_as_of_cutoff_time"] = str(identity["as_of_cutoff_time"])
    rankings["rating_bundle_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return rankings.sort_values("player_id", kind="stable").reset_index(drop=True)


def _promoted_bundle_history(
    root: Path,
    *,
    latest_bundle: Path,
    latest_manifest: dict[str, Any],
    explicit_bundle: bool,
) -> list[Path]:
    if explicit_bundle:
        return [latest_bundle]
    candidates: dict[str, tuple[str, Path]] = {}
    pointer_paths = list(data_io.glob(root / "pointer_history", "*.json"))
    pointer_paths.extend(path for path in (root / "latest.json",) if data_io.is_file(path))
    for pointer_path in pointer_paths:
        pointer = json.loads(data_io.read_text(pointer_path, encoding="utf-8"))
        bundle = resolve_pointer(root, pointer)
        manifest = verify_serving_bundle(bundle)
        identity = manifest["identity"]
        if identity["season"] != latest_manifest["identity"]["season"]:
            continue
        if str(identity["as_of_cutoff_time"]) > str(
            latest_manifest["identity"]["as_of_cutoff_time"]
        ):
            continue
        evidence_date = str(identity["evidence_through_date"])
        promoted_at = str(pointer.get("promoted_at", ""))
        previous = candidates.get(evidence_date)
        if previous is None or promoted_at >= previous[0]:
            candidates[evidence_date] = (promoted_at, bundle)
    if str(latest_manifest["identity"]["evidence_through_date"]) not in candidates:
        candidates[str(latest_manifest["identity"]["evidence_through_date"])] = (
            "",
            latest_bundle,
        )
    return [candidates[day][1] for day in sorted(candidates)]


def _rating_trajectories(
    rankings: pd.DataFrame,
    history_paths: list[Path],
    *,
    current_bundle_sha: str,
) -> dict[int, list[dict[str, Any]]]:
    baseline_by_player = dict(
        zip(
            rankings["player_id"].astype(int),
            rankings["preseason_nail"].astype(float),
            strict=True,
        )
    )
    trajectories: dict[int, list[dict[str, Any]]] = {
        player_id: [
            {
                "source": "preseason",
                "evidence_through_date": None,
                "as_of_cutoff_time": None,
                "rating": rating,
                "preseason_rating": rating,
                "in_season_delta": 0.0,
                "raw_on_court_possessions": 0.0,
                "evidence_weight": 0.0,
                "posterior_standard_error": None,
                "bundle_sha256": None,
            }
        ]
        for player_id, rating in baseline_by_player.items()
    }
    last_cutoff = ""
    for bundle in history_paths:
        manifest = verify_serving_bundle(bundle)
        identity = manifest["identity"]
        cutoff = str(identity["as_of_cutoff_time"])
        if cutoff <= last_cutoff:
            raise ValueError("Promoted serving trajectory is not strictly chronological")
        last_cutoff = cutoff
        state = data_io.parquet(bundle / SERVING_STATE_FILENAME)
        for row in state.itertuples(index=False):
            player_id = int(row.player_id)
            if player_id not in trajectories:
                continue
            baseline = baseline_by_player[player_id]
            if not np.isclose(float(row.frozen_preseason_nail), baseline, atol=1e-10):
                raise ValueError("A promoted trajectory changed its frozen preseason baseline")
            trajectories[player_id].append(
                {
                    "source": "in_season",
                    "evidence_through_date": str(identity["evidence_through_date"]),
                    "as_of_cutoff_time": cutoff,
                    "rating": float(row.updated_nail),
                    "preseason_rating": baseline,
                    "in_season_delta": float(row.current_season_residual),
                    "raw_on_court_possessions": float(row.raw_on_court_possessions),
                    "evidence_weight": float(row.evidence_weight),
                    "posterior_standard_error": float(row.posterior_standard_error),
                    "bundle_sha256": str(manifest["bundle_sha256"]),
                    "is_current": str(manifest["bundle_sha256"]) == current_bundle_sha,
                }
            )
    return trajectories


def _player_game_rating_trajectories(
    rankings: pd.DataFrame,
    trajectory: pd.DataFrame,
    *,
    current_bundle_sha: str,
    evidence_through_date: str,
    as_of_cutoff_time: str,
) -> dict[int, list[dict[str, Any]]]:
    """Build player-page trajectories from immutable postgame updater states."""

    trajectories: dict[int, list[dict[str, Any]]] = {}
    current_by_player = rankings.assign(
        player_id=rankings["player_id"].astype(int)
    ).set_index("player_id")
    grouped = {
        int(player_id): group.sort_values("game_number", kind="stable")
        for player_id, group in trajectory.groupby("player_id", sort=False)
    }
    for player_id, current in current_by_player.iterrows():
        baseline = float(current["preseason_nail"])
        points: list[dict[str, Any]] = [
            {
                "source": "preseason",
                "point_kind": "preseason",
                "evidence_through_date": None,
                "as_of_cutoff_time": None,
                "rating": baseline,
                "preseason_rating": baseline,
                "in_season_delta": 0.0,
                "raw_on_court_possessions": 0.0,
                "evidence_weight": 0.0,
                "posterior_standard_error": None,
                "bundle_sha256": None,
            }
        ]
        player_games = grouped.get(int(player_id), pd.DataFrame())
        for row in player_games.itertuples(index=False):
            points.append(
                {
                    "source": "in_season",
                    "point_kind": "postgame" if bool(row.played) else "skipped_game",
                    "game_number": int(row.game_number),
                    "appearance_number": (
                        int(row.appearance_number)
                        if pd.notna(row.appearance_number)
                        else None
                    ),
                    "game_id": str(row.game_id),
                    "game_date": str(row.game_date),
                    "team": str(row.team),
                    "opponent": str(row.opponent),
                    "played": bool(row.played),
                    # Preserve recorded zero-minute opportunities as well as
                    # appearances. Older trajectory formats may lack this field;
                    # their missing values are never invented by the API.
                    **({'actual_minutes': float(row.actual_minutes)}
                       if hasattr(row, 'actual_minutes') else {}),
                    "pregame_status": (
                        str(row.pregame_status)
                        if pd.notna(row.pregame_status)
                        else None
                    ),
                    "pregame_reason": (
                        str(row.pregame_reason)
                        if pd.notna(row.pregame_reason)
                        else None
                    ),
                    "evidence_through_date": str(row.game_date),
                    "as_of_cutoff_time": str(row.game_time_utc),
                    "rating": float(row.updated_rating),
                    # This slate's stored change, not the difference since the
                    # player's previous appearance (other slates can move ratings).
                    "rating_delta": (
                        float(row.rating_delta)
                        if pd.notna(getattr(row, "rating_delta", None)) else None
                    ),
                    "preseason_rating": float(row.preseason_rating),
                    "in_season_delta": float(row.in_season_delta),
                    "game_on_court_possessions": float(
                        row.game_on_court_possessions
                    ),
                    "raw_on_court_possessions": float(
                        row.cumulative_on_court_possessions
                    ),
                    "evidence_weight": float(row.evidence_weight),
                    "posterior_standard_error": None,
                    "bundle_sha256": current_bundle_sha,
                }
            )

        current_rating = float(current["rapm"])
        current_possessions = float(current["current_season_possessions"])
        last = points[-1]
        matches_latest_game = (
            last["point_kind"] in {"postgame", "skipped_game"}
            and np.isclose(float(last["rating"]), current_rating, atol=1e-10)
            and np.isclose(
                float(last["raw_on_court_possessions"]),
                current_possessions,
                atol=1e-10,
            )
        )
        if matches_latest_game:
            last["is_current"] = True
            last["posterior_standard_error"] = float(
                current["rating_posterior_standard_error"]
            )
        elif current_possessions > 0 or len(points) > 1:
            points.append(
                {
                    "source": "in_season",
                    "point_kind": "current",
                    "game_number": (
                        int(points[-1]["game_number"])
                        if points[-1].get("game_number") is not None
                        else None
                    ),
                    "evidence_through_date": evidence_through_date,
                    "as_of_cutoff_time": as_of_cutoff_time,
                    "rating": current_rating,
                    "preseason_rating": baseline,
                    "in_season_delta": float(current["in_season_delta"]),
                    "raw_on_court_possessions": current_possessions,
                    "evidence_weight": float(current["rating_evidence_weight"]),
                    "posterior_standard_error": float(
                        current["rating_posterior_standard_error"]
                    ),
                    "bundle_sha256": current_bundle_sha,
                    "is_current": True,
                }
            )
        trajectories[int(player_id)] = points
    return trajectories
