"""Reconcile current NBA rosters and extend frozen in-season model vocabularies.

The preseason ratings and minutes values remain frozen for every player that
was present at opening night.  A player first observed on an in-season roster
is appended with the same forward/cold-start machinery used by the preseason
build.  The catalog is monotone for the season so a later waiver cannot remove
identity or invalidate already-published game contributions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from nba_lineup_model.modeling.forward_portable_matchup_contextual_rapm import (
    _recover_replacement_tokens,
)
from nba_lineup_model.modeling.gap_returner_prior import (
    build_centered_value_conditioned_aging_gap_returner_priors,
)
from nba_lineup_model.modeling.prior_rapm import PRIOR_MEAN_COLUMN
from nba_lineup_model.players.opening_rosters import (
    PLAYER_MOVEMENT_URL,
    _movement_frame,
)
from nba_lineup_model.players.rosters import TEAM_ABBREVIATIONS
from nba_lineup_model.web_api.preseason_rankings_cache import (
    _completed_results,
    _exposure_history,
    _published_run,
    append_target_bios,
)

SEASON = "2026-27"
COMPLETED_SEASON = "2025-26"
ROSTER_COLUMNS = (
    "season",
    "team_id",
    "team_abbreviation",
    "player_id",
    "player_name",
    "listed_position",
    "height_raw",
    "height_inches",
    "weight_pounds",
    "birth_date",
    "age",
    "experience",
    "school",
    "how_acquired",
    "supplemental_status",
)


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def normalize_player_name(value: object) -> str:
    """Return a suffix-insensitive identity key for cross-source matching."""

    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+(?:Jr\.?|Sr\.?|II|III|IV|V)$", "", text, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def fetch_json(url: str, *, timeout_seconds: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.nba.com/",
            "User-Agent": "Mozilla/5.0",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"Roster source returned a non-object payload: {url}")
    return payload




def fetch_player_movements() -> dict[str, Any]:
    return fetch_json(PLAYER_MOVEMENT_URL)




def _unique_name_map(frame: pd.DataFrame) -> dict[str, int]:
    candidates: dict[str, set[int]] = {}
    for row in frame.loc[:, ["player_id", "player_name"]].dropna().itertuples(index=False):
        key = normalize_player_name(row.player_name)
        candidates.setdefault(key, set()).add(int(row.player_id))
    return {
        key: next(iter(player_ids))
        for key, player_ids in candidates.items()
        if key and len(player_ids) == 1
    }


def _movement_name_map(payload: Mapping[str, Any], path: Path) -> dict[str, int]:
    _atomic_write_bytes(path, _canonical_json_bytes(payload))
    frame = _movement_frame(path)
    candidates: dict[str, set[int]] = {}
    for row in frame.itertuples(index=False):
        key = normalize_player_name(str(row.player_slug).replace("-", " "))
        candidates.setdefault(key, set()).add(int(row.player_id))
    return {
        key: next(iter(player_ids))
        for key, player_ids in candidates.items()
        if key and len(player_ids) == 1
    }




def fetch_current_nba_rosters() -> dict[str, dict[str, Any]]:
    """Read canonical CommonTeamRoster responses; failures have no fallback."""
    from tempfile import TemporaryDirectory
    from nba_lineup_model.players.source import PlayerStatsCache, PlayerStatsClient

    with TemporaryDirectory(prefix="gestalt-nba-rosters-") as directory:
        client = PlayerStatsClient(cache=PlayerStatsCache(directory))
        try:
            return {
                team: client.fetch_team_roster(SEASON, team_id, use_cache=False).payload
                for team_id, team in sorted(TEAM_ABBREVIATIONS.items())
            }
        finally:
            client.close()


def nba_roster_frame(payloads: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Normalize official team rosters without a secondary identity crosswalk."""
    from nba_lineup_model.players.rosters import team_roster_frame

    expected = set(TEAM_ABBREVIATIONS.values())
    if set(payloads) != expected:
        raise ValueError("Official NBA roster snapshot must cover all 30 teams")
    teams = {team: team_id for team_id, team in TEAM_ABBREVIATIONS.items()}
    frames = [
        team_roster_frame(dict(payloads[team]), season=SEASON, team_id=teams[team])
        for team in sorted(payloads)
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty or frame['player_id'].duplicated().any():
        raise ValueError("Official NBA roster is empty or duplicates player IDs")
    return frame


def reconcile_current_roster(
    nba: pd.DataFrame,
    *,
    frozen_roster: pd.DataFrame,
    historical_panel: pd.DataFrame,
    movement_name_to_id: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use NBA's canonical IDs; history supplies no identity substitutions."""
    current = nba.copy()
    current['player_id'] = pd.to_numeric(current['player_id'], errors='raise').astype(int)
    if current.empty or (current['player_id'] <= 0).any() or current['player_id'].duplicated().any():
        raise ValueError("Official NBA roster contains invalid player identities")
    valid_teams = {str(key): value for key, value in TEAM_ABBREVIATIONS.items()}
    ids = current['team_id'].astype(str)
    if not ids.isin(valid_teams).all():
        raise ValueError("Official NBA roster contains an unknown team ID")
    current['team_abbreviation'] = ids.map(valid_teams)
    current['identity_source'] = 'nba_commonteamroster'
    current['supplemental_status'] = current.get('supplemental_status', '0')
    missing = set(ROSTER_COLUMNS) - set(current.columns)
    if missing:
        raise ValueError(f"Official NBA roster missing fields: {sorted(missing)}")
    unresolved = pd.DataFrame(columns=['team', 'player_id', 'player_name', 'reason'])
    return current.sort_values(['team_abbreviation', 'player_name']).reset_index(drop=True), unresolved


def _single_match(root: Path, pattern: str, label: str) -> Path:
    paths = sorted(root.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}")
    return paths[0]


def _extend_preseason_catalogs(
    *,
    base_rankings_path: Path,
    base_profiles_path: Path,
    current_roster: pd.DataFrame,
    panel_path: Path,
    historical_profiles_path: Path,
    rankings_output_path: Path,
    profiles_output_path: Path,
) -> list[int]:
    """Append roster additions without rebuilding or rewriting frozen states.

    The deployed release intentionally excludes the raw multi-decade profile
    inputs used by the offline preseason build.  Runtime reconciliation uses
    the already-fitted forward prior machinery for historical players and the
    frozen replacement prior for true cold starts.  A last materialized profile
    is retained for display/provenance, but it is not rescored into the prior.
    """

    rankings = pd.read_parquet(base_rankings_path).copy()
    profiles = pd.read_parquet(base_profiles_path).copy()
    for label, frame in (("rankings", rankings), ("profiles", profiles)):
        if frame.empty or frame["player_id"].duplicated().any():
            raise ValueError(f"Frozen preseason {label} catalog is invalid")
    current = current_roster.copy()
    current["player_id"] = pd.to_numeric(
        current["player_id"], errors="raise"
    ).astype(int)
    current_by_id = current.set_index("player_id")
    additions = sorted(set(current_by_id.index) - set(rankings["player_id"].astype(int)))

    panel = pd.read_parquet(panel_path)
    latest_panel = (
        panel.sort_values(["player_id", "season_start_year"], kind="stable")
        .drop_duplicates("player_id", keep="last")
        .set_index("player_id")
    )
    forward_prior: dict[int, float] = {}
    prior_alignment = 0.0
    if additions:
        target_panel = append_target_bios(
            panel,
            current,
            target_season=SEASON,
            allow_missing_draft_history=True,
        )
        run_dir, run_id = _published_run(COMPLETED_SEASON)
        historical = pd.read_parquet(run_dir / "historical_player_coefficients.parquet")
        results = _completed_results(historical)
        exposure_history = _exposure_history(run_id, historical, panel)
        prior_metadata = pd.read_parquet(
            run_dir / "season_player_prior_metadata.parquet"
        )
        replacement_tokens = _recover_replacement_tokens(
            prior_metadata,
            tuple(result.season for result in results),
        )
        priors, _contract = (
            build_centered_value_conditioned_aging_gap_returner_priors(
                season=SEASON,
                panel=target_panel,
                completed_results=results,
                exposure_history=exposure_history,
                replacement_tokens=replacement_tokens,
            )
        )
        forward_prior = dict(
            zip(
                priors["player_id"].astype(int),
                priors[PRIOR_MEAN_COLUMN].astype(float),
                strict=True,
            )
        )
        aligned = rankings.loc[
            rankings["forecast_source"].astype(str).str.startswith("forward_")
            & rankings["player_id"].astype(int).isin(forward_prior),
            ["player_id", "prior_rating"],
        ].copy()
        aligned["candidate"] = aligned["player_id"].astype(int).map(forward_prior)
        differences = pd.to_numeric(
            aligned["prior_rating"], errors="coerce"
        ) - pd.to_numeric(aligned["candidate"], errors="coerce")
        differences = differences[np.isfinite(differences)]
        if differences.empty:
            raise ValueError("Cannot align appended player priors to the frozen catalog")
        prior_alignment = float(differences.median())

    replacement = rankings.loc[
        rankings["forecast_source"].eq("replacement_cold_start_prior"), "rapm"
    ]
    replacement_rating = float(pd.to_numeric(replacement, errors="coerce").median())
    if not np.isfinite(replacement_rating):
        raise ValueError("Frozen replacement cold-start prior is unavailable")

    ranking_rows: list[dict[str, object]] = []
    for player_id in additions:
        roster = current_by_id.loc[player_id]
        historical = (
            latest_panel.loc[player_id]
            if player_id in latest_panel.index
            else pd.Series(dtype=object)
        )
        has_forward_prior = player_id in forward_prior
        prior_rating = (
            float(forward_prior[player_id]) + prior_alignment
            if has_forward_prior
            else replacement_rating
        )
        row = {column: pd.NA for column in rankings.columns}
        row.update(
            {
                "season": SEASON,
                "player_id": player_id,
                "player_name": str(roster["player_name"]),
                "team": str(roster["team_abbreviation"]),
                "position": str(roster["listed_position"]),
                "draft_year": historical.get("draft_year", pd.NA),
                "draft_round": historical.get("draft_round", pd.NA),
                "draft_number": historical.get("draft_number", pd.NA),
                "is_undrafted": historical.get("is_undrafted", True),
                "draft_class_year": historical.get("draft_year", pd.NA),
                "rapm": prior_rating,
                "prior_rating": prior_rating,
                "season_update": np.nan,
                "additive_profile_adjustment": np.nan,
                "observed_context_exposure": np.nan,
                "possessions": 0.0,
                "games": 0,
                "age": float(roster["age"]),
                "rookie_season": (
                    SEASON if not has_forward_prior else pd.NA
                ),
                "profile_source": (
                    "in_season_historical_forward_prior"
                    if has_forward_prior
                    else "in_season_replacement_cold_start_prior"
                ),
                "forecast_source": (
                    "in_season_historical_forward_prior"
                    if has_forward_prior
                    else "in_season_replacement_cold_start_prior"
                ),
                "additive_profile_breakdown_json": None,
            }
        )
        ranking_rows.append(row)
    if ranking_rows:
        rankings = pd.concat(
            [rankings, pd.DataFrame(ranking_rows, columns=rankings.columns)],
            ignore_index=True,
        )

    active_team = current_by_id["team_abbreviation"].astype(str).to_dict()
    active_name = current_by_id["player_name"].astype(str).to_dict()
    active_position = current_by_id["listed_position"].astype(str).to_dict()
    rankings["team"] = rankings.apply(
        lambda row: active_team.get(int(row["player_id"]), str(row["team"])), axis=1
    )
    rankings["player_name"] = rankings.apply(
        lambda row: active_name.get(int(row["player_id"]), str(row["player_name"])),
        axis=1,
    )
    rankings["position"] = rankings.apply(
        lambda row: active_position.get(int(row["player_id"]), str(row["position"])),
        axis=1,
    )
    rankings = rankings.sort_values("player_id", kind="stable").reset_index(drop=True)
    rankings.to_parquet(rankings_output_path, index=False)

    historical_profiles = pd.read_parquet(historical_profiles_path)
    historical_profiles["player_id"] = pd.to_numeric(
        historical_profiles["player_id"], errors="raise"
    ).astype(int)
    generic_profile = profiles.select_dtypes(include="number").median(numeric_only=True)
    profile_rows: list[dict[str, object]] = []
    for player_id in additions:
        roster = current_by_id.loc[player_id]
        history = historical_profiles.loc[
            historical_profiles["player_id"].eq(player_id)
        ].sort_values("season", kind="stable")
        if history.empty:
            row = {column: pd.NA for column in profiles.columns}
            for column, value in generic_profile.items():
                row[column] = value
            source = "runtime_replacement_profile"
            imputed = 1
            replacement_weight = 1.0
        else:
            row = history.iloc[-1].to_dict()
            source = "last_materialized_profile_runtime_padding"
            imputed = int(row.get("profile_imputed", 0))
            replacement_weight = float(row.get("profile_replacement_weight", 0.0))
        row.update(
            {
                "target_season": SEASON,
                "season": SEASON,
                "player_id": player_id,
                "player_name": str(roster["player_name"]),
                "is_rookie": player_id not in latest_panel.index,
                "profile_source": source,
                "profile_imputed": imputed,
                "profile_replacement_weight": replacement_weight,
            }
        )
        profile_rows.append({column: row.get(column, pd.NA) for column in profiles.columns})
    if profile_rows:
        profiles = pd.concat(
            [profiles, pd.DataFrame(profile_rows, columns=profiles.columns)],
            ignore_index=True,
        )
    profiles = profiles.sort_values("player_id", kind="stable").reset_index(drop=True)
    profiles.to_parquet(profiles_output_path, index=False)
    metadata_path = rankings_output_path.with_suffix(".json")
    source_metadata_path = base_rankings_path.with_suffix(".json")
    metadata = (
        json.loads(source_metadata_path.read_text())
        if source_metadata_path.is_file()
        else {}
    )
    metadata.update(
        {
            "frozen_catalog_extended": True,
            "player_count": len(rankings),
            "added_player_ids": additions,
            "runtime_contract": (
                "incumbent preseason states are immutable; historical additions "
                "receive an aligned fitted forward prior; true cold starts receive "
                "the frozen replacement prior"
            ),
        }
    )
    _atomic_write_bytes(metadata_path, _canonical_json_bytes(metadata))
    return additions


def _merge_minutes_catalog(
    *,
    base_cache_path: Path,
    prior_overlay_path: Path | None,
    candidate: dict[str, object],
    current_roster: pd.DataFrame,
    output_path: Path,
) -> list[int]:
    source = (
        prior_overlay_path
        if prior_overlay_path and prior_overlay_path.is_file()
        else base_cache_path
    )
    base_payload = json.loads(source.read_text())
    base_minutes = base_payload.get("minutes", base_payload)
    if not isinstance(base_minutes, dict) or not isinstance(base_minutes.get("players"), list):
        raise ValueError("Frozen minutes cache has no player catalog")
    candidate_rows = {
        int(row["player_id"]): row for row in candidate.get("players", [])
    }
    base_rows = {
        int(row["player_id"]): dict(row) for row in base_minutes["players"]
    }
    active = current_roster.copy()
    active["player_id"] = pd.to_numeric(active["player_id"], errors="raise").astype(int)
    active_by_id = active.set_index("player_id")
    additions = sorted(set(candidate_rows) - set(base_rows))
    for player_id in additions:
        base_rows[player_id] = dict(candidate_rows[player_id])
    for player_id, roster_row in active_by_id.iterrows():
        if int(player_id) not in base_rows:
            continue
        base_rows[int(player_id)]["team"] = str(roster_row["team_abbreviation"])
        base_rows[int(player_id)]["team_id"] = int(roster_row["team_id"])
        base_rows[int(player_id)]["player_name"] = str(roster_row["player_name"])
    merged = dict(base_minutes)
    merged["players"] = sorted(
        base_rows.values(), key=lambda row: (str(row.get("team", "")), int(row["player_id"]))
    )
    _atomic_write_bytes(output_path, _canonical_json_bytes({"minutes": merged}))
    return additions


def _extend_runtime_minutes_catalog(
    *,
    base_cache_path: Path,
    prior_overlay_path: Path | None,
    rankings: pd.DataFrame,
    current_roster: pd.DataFrame,
    rotation_history_path: Path,
    output_path: Path,
) -> list[int]:
    """Append only missing minute priors from materialized rotation history."""

    source = (
        prior_overlay_path
        if prior_overlay_path and prior_overlay_path.is_file()
        else base_cache_path
    )
    payload = json.loads(source.read_text())
    minutes = payload.get("minutes", payload)
    if not isinstance(minutes, dict) or not isinstance(minutes.get("players"), list):
        raise ValueError("Frozen minutes cache has no player catalog")
    rows = {int(row["player_id"]): dict(row) for row in minutes["players"]}
    current = current_roster.copy()
    current["player_id"] = pd.to_numeric(
        current["player_id"], errors="raise"
    ).astype(int)
    current_by_id = current.set_index("player_id")
    ratings = rankings.copy()
    ratings["player_id"] = pd.to_numeric(
        ratings["player_id"], errors="raise"
    ).astype(int)
    rating_by_id = ratings.set_index("player_id")
    additions = sorted(set(current_by_id.index) - set(rows))
    history = pd.read_parquet(rotation_history_path)
    history["player_id"] = pd.to_numeric(
        history["player_id"], errors="raise"
    ).astype(int)
    cold_start_rows = [
        row
        for row in rows.values()
        if bool(row.get("uses_draft_cold_start_prior"))
        or not bool(row.get("has_prior_minutes_state", True))
    ]
    cold_start_conditional = float(
        pd.to_numeric(
            pd.Series(
                [row.get("conditional_minutes_per_game") for row in cold_start_rows]
            ),
            errors="coerce",
        ).median()
    )
    if not np.isfinite(cold_start_conditional) or cold_start_conditional <= 0:
        cold_start_conditional = 8.0
    for player_id in additions:
        roster = current_by_id.loc[player_id]
        player_history = history.loc[history["player_id"].eq(player_id)].sort_values(
            ["season_start_year", "season"], kind="stable"
        )
        if player_history.empty:
            conditional = cold_start_conditional
            prior_minutes = 0.0
            has_prior = False
            rating_source = "in-season replacement cold-start prior"
        else:
            latest = player_history.iloc[-1]
            conditional = float(latest["actual_minutes_per_available_game"])
            prior_minutes = float(latest["actual_total_minutes"])
            has_prior = np.isfinite(conditional) and conditional >= 0.0
            if not has_prior:
                conditional = cold_start_conditional
                prior_minutes = 0.0
            rating_source = "in-season last observed available-game role"
        rating = rating_by_id.loc[player_id]
        template = {key: None for key in next(iter(rows.values()))}
        template.update(
            {
                "player_id": player_id,
                "player_name": str(roster["player_name"]),
                "position": str(roster["listed_position"]),
                "age": float(roster["age"]),
                "team": str(roster["team_abbreviation"]),
                "prior_season_minutes": prior_minutes,
                "is_rating_fallback": False,
                "is_baseline_rotation_candidate": conditional > 0.0,
                "has_prior_availability_state": has_prior,
                "has_prior_minutes_state": has_prior,
                "availability_probability": 1.0,
                "conditional_minutes_per_game": conditional,
                "raw_expected_total_minutes": 82.0 * conditional,
                # Frozen normalized team allocations are not rewritten when a
                # player signs. The in-season forecaster normalizes the live
                # conditional-role state to 240 minutes at each game cutoff.
                "baseline_minute_share": 0.0,
                "baseline_minutes_per_game": 0.0,
                "projected_nail": float(rating["rapm"]),
                "rating_source": rating_source,
                "model_availability_probability": 1.0,
                "injury_adjusted": False,
                "injury_expected_games_missed": 0.0,
                "uses_draft_cold_start_prior": not has_prior,
                "uses_incumbent_team_strength": False,
                "is_incumbent_team_strength_incumbent": False,
            }
        )
        rows[player_id] = template
    for player_id, roster in current_by_id.iterrows():
        if int(player_id) not in rows:
            continue
        rows[int(player_id)]["team"] = str(roster["team_abbreviation"])
        rows[int(player_id)]["player_name"] = str(roster["player_name"])
        rows[int(player_id)]["position"] = str(roster["listed_position"])
    merged = dict(minutes)
    merged["players"] = sorted(
        rows.values(), key=lambda row: (str(row.get("team", "")), int(row["player_id"]))
    )
    _atomic_write_bytes(output_path, _canonical_json_bytes({"minutes": merged}))
    return additions


def _validate_coupled_preseason_prior_catalogs(
    *,
    rankings_path: Path,
    minutes_path: Path,
) -> None:
    """Require ratings and minutes priors to share one immutable player universe."""

    ranking_ids = set(
        pd.to_numeric(
            pd.read_parquet(rankings_path, columns=["player_id"])["player_id"],
            errors="raise",
        ).astype(int)
    )
    payload = json.loads(minutes_path.read_text())
    minutes = payload.get("minutes", payload)
    players = minutes.get("players") if isinstance(minutes, dict) else None
    if not isinstance(players, list):
        raise ValueError("Frozen minutes cache has no player catalog")
    minutes_frame = pd.DataFrame(players)
    if "player_id" not in minutes_frame or minutes_frame["player_id"].duplicated().any():
        raise ValueError("Frozen minutes cache has an invalid player catalog")
    minutes_ids = set(
        pd.to_numeric(minutes_frame["player_id"], errors="raise").astype(int)
    )
    if ranking_ids != minutes_ids:
        raise ValueError(
            "Coupled preseason prior player universe mismatch: "
            f"ratings_only={sorted(ranking_ids - minutes_ids)}; "
            f"minutes_only={sorted(minutes_ids - ranking_ids)}"
        )


def run(**configuration) -> dict[str, object]:
    """Run the durable reconciliation producer from the bound daily job.

    Legacy direct writers cannot provide the required environment, accepted
    model, pinned inputs, or publication transaction and fail closed.
    """
    from nba_lineup_model.players.reconciliation_workflow import run as reconcile
    return reconcile(**configuration)



def main() -> None:
    raise RuntimeError('Use the environment-bound reconcile_in_season_rosters Dagster job; the legacy local-overlay CLI is retired')


if __name__ == "__main__":
    main()
