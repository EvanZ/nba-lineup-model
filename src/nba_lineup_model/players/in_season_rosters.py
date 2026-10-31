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
from nba_lineup_model.web_api.preseason_minutes import (
    build_forward_conditional_preseason_minutes_payload,
)
from nba_lineup_model.web_api.preseason_rankings_cache import (
    _completed_results,
    _exposure_history,
    _published_run,
    append_target_bios,
)

SEASON = "2026-27"
COMPLETED_SEASON = "2025-26"
ESPN_TEAMS_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
)
ESPN_ROSTER_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}/roster"
)
ESPN_TEAM_ALIASES = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UT": "UTA",
    "UTAH": "UTA",
    "WSH": "WAS",
}
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


def fetch_current_espn_rosters() -> dict[str, dict[str, Any]]:
    teams_payload = fetch_json(ESPN_TEAMS_URL)
    teams: list[tuple[str, str]] = []
    for sport in teams_payload.get("sports", []):
        for league in sport.get("leagues", []) if isinstance(sport, dict) else []:
            for wrapper in league.get("teams", []) if isinstance(league, dict) else []:
                team = wrapper.get("team", {}) if isinstance(wrapper, dict) else {}
                team_id = str(team.get("id", ""))
                abbreviation = ESPN_TEAM_ALIASES.get(
                    str(team.get("abbreviation", "")).upper(),
                    str(team.get("abbreviation", "")).upper(),
                )
                if team_id and abbreviation:
                    teams.append((team_id, abbreviation))
    if len(teams) != 30 or len({team for _, team in teams}) != 30:
        raise ValueError(f"ESPN team index returned {len(teams)} unique NBA teams")
    return {
        team: fetch_json(ESPN_ROSTER_URL.format(team_id=team_id))
        for team_id, team in sorted(teams, key=lambda item: item[1])
    }


def fetch_player_movements() -> dict[str, Any]:
    return fetch_json(PLAYER_MOVEMENT_URL)


def espn_roster_frame(payloads: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for raw_team, payload in sorted(payloads.items()):
        team = ESPN_TEAM_ALIASES.get(str(raw_team).upper(), str(raw_team).upper())
        for athlete in payload.get("athletes", []):
            if not isinstance(athlete, dict):
                continue
            position = athlete.get("position", {})
            college = athlete.get("college", {})
            experience = athlete.get("experience", {})
            status = athlete.get("status", {})
            rows.append(
                {
                    "team": team,
                    "espn_player_id": str(athlete.get("id", "")),
                    "player_name": str(
                        athlete.get("fullName") or athlete.get("displayName") or ""
                    ).strip(),
                    "listed_position": str(
                        position.get("abbreviation", "")
                        if isinstance(position, dict)
                        else ""
                    ),
                    "height_inches": pd.to_numeric(athlete.get("height"), errors="coerce"),
                    "height_raw": str(athlete.get("displayHeight", "")),
                    "weight_pounds": pd.to_numeric(athlete.get("weight"), errors="coerce"),
                    "birth_date": str(athlete.get("dateOfBirth", "")),
                    "age": pd.to_numeric(athlete.get("age"), errors="coerce"),
                    "experience": str(
                        experience.get("years", "")
                        if isinstance(experience, dict)
                        else ""
                    ),
                    "school": str(
                        college.get("name", "") if isinstance(college, dict) else ""
                    ),
                    "espn_status": str(
                        status.get("type", "") if isinstance(status, dict) else ""
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["player_name"].eq("").any():
        raise ValueError("ESPN roster snapshot has missing player identity")
    if frame.duplicated(["team", "espn_player_id"]).any():
        raise ValueError("ESPN roster snapshot duplicates a team/player record")
    return frame.sort_values(["team", "player_name"], kind="stable").reset_index(drop=True)


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


def reconcile_current_roster(
    espn: pd.DataFrame,
    *,
    frozen_roster: pd.DataFrame,
    historical_panel: pd.DataFrame,
    movement_name_to_id: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve an ESPN current-roster snapshot to stable NBA player IDs."""

    frozen = frozen_roster.copy()
    frozen["player_id"] = pd.to_numeric(frozen["player_id"], errors="raise").astype(int)
    history = historical_panel.copy()
    history["player_id"] = pd.to_numeric(history["player_id"], errors="raise").astype(int)
    identity_sources = (
        ("frozen_roster", _unique_name_map(frozen)),
        ("historical_panel", _unique_name_map(history)),
        ("player_movement", dict(movement_name_to_id)),
    )
    frozen_by_id = frozen.sort_values("player_id", kind="stable").drop_duplicates(
        "player_id", keep="last"
    ).set_index("player_id")
    historical_by_id = history.sort_values(
        ["player_id", "season_start_year"], kind="stable"
    ).drop_duplicates("player_id", keep="last").set_index("player_id")
    team_id_by_abbreviation = {
        abbreviation: team_id for team_id, abbreviation in TEAM_ABBREVIATIONS.items()
    }
    resolved_rows: list[dict[str, object]] = []
    unresolved_rows: list[dict[str, object]] = []
    for row in espn.itertuples(index=False):
        key = normalize_player_name(row.player_name)
        candidates = {
            int(mapping[key])
            for _source, mapping in identity_sources
            if key in mapping
        }
        if len(candidates) != 1:
            unresolved_rows.append(
                {
                    "team": row.team,
                    "espn_player_id": row.espn_player_id,
                    "player_name": row.player_name,
                    "candidate_player_ids_json": json.dumps(sorted(candidates)),
                    "reason": "no stable NBA identity" if not candidates else "ambiguous identity",
                }
            )
            continue
        player_id = next(iter(candidates))
        identity_source = next(
            source for source, mapping in identity_sources if mapping.get(key) == player_id
        )
        if player_id in frozen_by_id.index:
            base = frozen_by_id.loc[player_id].to_dict()
        elif player_id in historical_by_id.index:
            historical = historical_by_id.loc[player_id]
            base = {
                "listed_position": historical.get("listed_position", ""),
                "height_inches": historical.get("height_inches"),
                "weight_pounds": historical.get("weight_pounds"),
                "age": historical.get("age"),
                "experience": historical.get("nba_experience_years", ""),
                "school": historical.get("college", ""),
            }
        else:
            base = {}
        team = str(row.team)
        height_inches = row.height_inches if pd.notna(row.height_inches) else base.get(
            "height_inches"
        )
        resolved_rows.append(
            {
                "season": SEASON,
                "team_id": str(team_id_by_abbreviation[team]),
                "team_abbreviation": team,
                "player_id": str(player_id),
                "player_name": str(row.player_name),
                "listed_position": str(row.listed_position or base.get("listed_position", "")),
                "height_raw": str(row.height_raw or base.get("height_raw", "")),
                "height_inches": int(height_inches) if pd.notna(height_inches) else 0,
                "weight_pounds": float(
                    row.weight_pounds
                    if pd.notna(row.weight_pounds)
                    else base.get("weight_pounds", 0.0)
                ),
                "birth_date": str(row.birth_date or base.get("birth_date", "")),
                "age": float(row.age if pd.notna(row.age) else base.get("age", 0.0)),
                "experience": str(row.experience or base.get("experience", "")),
                "school": str(row.school or base.get("school", "")),
                "how_acquired": str(
                    base.get("how_acquired", "In-season roster reconciliation")
                ),
                "supplemental_status": str(base.get("supplemental_status", "0")),
                "espn_player_id": str(row.espn_player_id),
                "espn_status": str(row.espn_status),
                "identity_source": identity_source,
            }
        )
    current = pd.DataFrame(resolved_rows)
    unresolved = pd.DataFrame(unresolved_rows)
    if current.empty or current["player_id"].duplicated().any():
        raise ValueError("Resolved current roster is empty or duplicates NBA player IDs")
    return (
        current.sort_values(["team_abbreviation", "player_name"], kind="stable")
        .reset_index(drop=True),
        unresolved.sort_values(["team", "player_name"], kind="stable").reset_index(
            drop=True
        )
        if not unresolved.empty
        else pd.DataFrame(
            columns=(
                "team",
                "espn_player_id",
                "player_name",
                "candidate_player_ids_json",
                "reason",
            )
        ),
    )


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


def run(
    *,
    release_root: Path,
    state_root: Path,
    espn_payloads: Mapping[str, Mapping[str, Any]] | None = None,
    movement_payload: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Poll roster sources and publish a monotone in-season prior overlay."""

    release_root = release_root.resolve()
    state_root.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    espn_payloads = dict(espn_payloads or fetch_current_espn_rosters())
    movement_payload = dict(movement_payload or fetch_player_movements())
    espn_bytes = _canonical_json_bytes(espn_payloads)
    movement_bytes = _canonical_json_bytes(movement_payload)
    espn_frame = espn_roster_frame(espn_payloads)
    fingerprint_columns = [
        column for column in espn_frame.columns if column != "espn_status"
    ]
    roster_fingerprint = _canonical_json_bytes(
        espn_frame.loc[:, fingerprint_columns].fillna("").to_dict(orient="records")
    )
    # ESPN embeds injuries and other volatile presentation fields in its
    # roster response. They are archived but deliberately excluded from the
    # idempotence key; only membership/identity/biography fields trigger work.
    source_digest = _sha256_bytes(roster_fingerprint + movement_bytes)
    latest_path = state_root / "latest.json"
    if latest_path.is_file() and not force:
        previous = json.loads(latest_path.read_text())
        if str(previous.get("source_digest")) == source_digest:
            return {**previous, "status": "unchanged", "changed": False}

    run_root = state_root / "runs" / f"source_sha256={source_digest}"
    run_root.mkdir(parents=True, exist_ok=True)
    espn_snapshot = state_root / "raw" / "espn_rosters" / f"{_sha256_bytes(espn_bytes)}.json"
    movement_snapshot = (
        state_root / "raw" / "nba_player_movement" / f"{_sha256_bytes(movement_bytes)}.json"
    )
    if not espn_snapshot.exists():
        _atomic_write_bytes(espn_snapshot, espn_bytes)
    if not movement_snapshot.exists():
        _atomic_write_bytes(movement_snapshot, movement_bytes)

    frozen_roster_path = release_root / "data/curated/team_rosters" / SEASON / "part-00000.parquet"
    panel_path = release_root / "data/analytical/player_season_panel/player_seasons.parquet"
    frozen_roster = pd.read_parquet(frozen_roster_path)
    historical_panel = pd.read_parquet(panel_path)
    movement_input_path = run_root / "movement-normalized-input.json"
    movement_name_to_id = _movement_name_map(movement_payload, movement_input_path)
    movements = _movement_frame(movement_input_path)
    current, unresolved = reconcile_current_roster(
        espn_frame,
        frozen_roster=frozen_roster,
        historical_panel=historical_panel,
        movement_name_to_id=movement_name_to_id,
    )
    roster_path = run_root / "current_roster.parquet"
    current.loc[:, ROSTER_COLUMNS].to_parquet(roster_path, index=False)
    identity_path = run_root / "roster_identity.parquet"
    current.to_parquet(identity_path, index=False)
    unresolved_path = run_root / "unresolved_players.parquet"
    unresolved.to_parquet(unresolved_path, index=False)

    frozen_rankings = _single_match(
        release_root,
        "artifacts/web/preseason_rankings/forward_nail_rapm_v1212_residualized_lambda/*/2026-27.parquet",
        "frozen preseason ranking",
    )
    frozen_profiles = frozen_rankings.parent / "2026-27-profiles.parquet"
    if not frozen_profiles.is_file():
        frozen_profiles = _single_match(
            release_root,
            "artifacts/web/preseason_profiles/forward_nail_rapm_v1212_residualized_lambda/*/2026-27.parquet",
            "frozen preseason profile",
        )
    previous: dict[str, Any] = {}
    if latest_path.is_file():
        previous = json.loads(latest_path.read_text())
    previous_artifacts = previous.get("artifacts", {}) if isinstance(previous, dict) else {}
    prior_rankings = Path(
        str(previous_artifacts.get("preseason_rankings", frozen_rankings))
    )
    prior_profiles = Path(
        str(previous_artifacts.get("preseason_profiles", frozen_profiles))
    )
    rankings_path = run_root / "preseason_rankings.parquet"
    profiles_path = run_root / "preseason_profiles.parquet"
    historical_profiles_path = _single_match(
        release_root,
        "artifacts/web/historical_profiles/forward_nail_rapm_v1212_residualized_lambda/*.parquet",
        "historical player profile",
    )
    old_cwd = Path.cwd()
    try:
        # Published model/data paths are release-relative by design.
        import os

        os.chdir(release_root)
        added_ids = _extend_preseason_catalogs(
            base_rankings_path=prior_rankings,
            base_profiles_path=prior_profiles,
            current_roster=current,
            panel_path=panel_path,
            historical_profiles_path=historical_profiles_path,
            rankings_output_path=rankings_path,
            profiles_output_path=profiles_path,
        )
        rankings = pd.read_parquet(rankings_path)
        candidate_minutes = build_forward_conditional_preseason_minutes_payload(
            preseason_rankings=rankings,
            roster_path=roster_path,
            target_season=SEASON,
        )
    finally:
        os.chdir(old_cwd)

    frozen_minutes_cache = _single_match(
        release_root,
        "artifacts/web/win_projections/*/*.json",
        "frozen minutes cache",
    )
    prior_minutes_value = previous_artifacts.get("preseason_minutes")
    prior_minutes = Path(str(prior_minutes_value)) if prior_minutes_value else None
    minutes_path = run_root / "preseason_minutes.json"
    minute_additions = _merge_minutes_catalog(
        base_cache_path=frozen_minutes_cache,
        prior_overlay_path=prior_minutes,
        candidate=candidate_minutes,
        current_roster=current,
        output_path=minutes_path,
    )
    frozen_ids = set(pd.read_parquet(frozen_rankings)["player_id"].astype(int))
    ranking_ids = set(pd.read_parquet(rankings_path)["player_id"].astype(int))
    if sorted(ranking_ids - frozen_ids) != added_ids:
        raise ValueError("Published roster additions do not match ranking catalog growth")
    added = current.loc[
        pd.to_numeric(current["player_id"], errors="raise").astype(int).isin(added_ids)
    ].copy()
    added["first_observed_at"] = fetched_at
    add_movements = movements.loc[movements["transaction_type"].ne("Waive")].copy()
    add_movements["player_id"] = pd.to_numeric(
        add_movements["player_id"], errors="raise"
    ).astype(int)
    latest_movement = (
        add_movements.sort_values(
            ["player_id", "team", "transaction_date", "additional_sort"], kind="stable"
        )
        .drop_duplicates(["player_id", "team"], keep="last")
        .set_index(["player_id", "team"])
    )
    transaction_dates: list[object] = []
    transaction_types: list[object] = []
    for row in added.itertuples(index=False):
        key = (int(row.player_id), str(row.team_abbreviation))
        if key in latest_movement.index:
            movement = latest_movement.loc[key]
            transaction_dates.append(movement["transaction_date"])
            transaction_types.append(str(movement["transaction_type"]))
        else:
            transaction_dates.append(pd.NA)
            transaction_types.append(pd.NA)
    added["transaction_date"] = transaction_dates
    added["transaction_type"] = transaction_types
    additions_path = run_root / "roster_additions.parquet"
    added.to_parquet(additions_path, index=False)

    artifacts = {
        "current_roster": str(roster_path),
        "roster_identity": str(identity_path),
        "roster_additions": str(additions_path),
        "unresolved_players": str(unresolved_path),
        "preseason_rankings": str(rankings_path),
        "preseason_profiles": str(profiles_path),
        "preseason_minutes": str(minutes_path),
        "espn_snapshot": str(espn_snapshot),
        "player_movement_snapshot": str(movement_snapshot),
    }
    artifact_sha256 = {
        label: _sha256_path(Path(path)) for label, path in artifacts.items()
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "season": SEASON,
        "status": "updated",
        "changed": True,
        "fetched_at": fetched_at,
        "source_digest": source_digest,
        "source_sha256": {
            "espn_rosters": _sha256_bytes(espn_bytes),
            "nba_player_movement": _sha256_bytes(movement_bytes),
        },
        "coverage": {
            "teams": int(current["team_abbreviation"].nunique()),
            "current_players": len(current),
            "catalog_additions": len(added_ids),
            "unresolved_players": len(unresolved),
        },
        "added_player_ids": added_ids,
        "minutes_added_player_ids": minute_additions,
        "artifacts": artifacts,
        "artifact_sha256": artifact_sha256,
    }
    manifest_path = run_root / "manifest.json"
    _atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))
    manifest["manifest_sha256"] = _sha256_path(manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    _atomic_write_bytes(latest_path, _canonical_json_bytes(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                release_root=args.release_root,
                state_root=args.state_root,
                force=args.force,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
