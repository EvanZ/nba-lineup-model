"""Build the current-season model chain from verified NBA daily data artifacts.

This module is deliberately separate from the legacy shadow monitors.  Its
runtime inputs are limited to (a) one content-addressed frozen-prior bundle and
(b) the verified ``daily_data`` chain.  It never reads Schedule finals, ESPN
box scores, or mutable current-season serving state.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

from nba_lineup_model.modeling.game_script_minutes import competitive_role_minutes
from nba_lineup_model.modeling.in_season_minutes import (
    COACH_DNP,
    G_LEAGUE_ASSIGNMENT,
    INACTIVE,
    MEDICAL_ABSENCE,
    PLAYED,
    UNKNOWN,
    InSeasonMinutesConfig,
    advance_minutes_state,
    forecast_team_minutes,
    initialize_minutes_state,
    regulation_equivalent_minutes,
)
from nba_lineup_model.modeling.in_season_minutes import (
    UPDATER_VERSION as MINUTES_UPDATER_VERSION,
)
from nba_lineup_model.modeling.in_season_nail import (
    InSeasonNailConfig,
    build_game_contribution,
    validate_player_state,
)
from nba_lineup_model.modeling.in_season_nail_monitor import _context_offsets
from nba_lineup_model.season.daily_artifact_common import canonical_json, file_sha256
from nba_lineup_model.season.daily_data_artifact import verify_daily_data_artifact
from nba_lineup_model.season.daily_model_artifact import (
    PublishedDailyModelArtifact,
    publish_daily_model_artifact,
    verify_daily_model_artifact,
)
from nba_lineup_model.web_api.inference import projected_win_pct

DEFAULT_SEASON = "2026-27"
DEFAULT_DATA_ROOT = Path("/var/lib/nba-gestalt-dev/daily-data-nba-v2")
DEFAULT_MODEL_ROOT = Path("/var/lib/nba-gestalt-dev/daily-model-nba-v1")
DEFAULT_MATCHUP_URL = "http://127.0.0.1:8010/api/matchups"
CORE_MODEL_VERSION = "forward_nail_rapm_v1212_residualized_lambda"
RATINGS_UPDATER_VERSION = "possession_residual_ridge_v1"
MODEL_VERSION = "official_nba_daily_model_v1"
G_LEAGUE_TOKENS = ("G LEAGUE", "G-LEAGUE", "GLEAGUE", "ASSIGNMENT", "TWO-WAY")
INACTIVE_TOKENS = ("SUSPENSION", "PERSONAL")
INJURY_STATUSES = frozenset({"OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE"})


class DailyModelReplayBlocked(RuntimeError):
    """Raised after a date-level failure report is retained for inspection."""


@dataclass(frozen=True)
class DailyModelReplayConfig:
    """Frozen updater settings whose canonical hash identifies one chain."""

    season: str = DEFAULT_SEASON
    model_version: str = MODEL_VERSION
    core_model_version: str = CORE_MODEL_VERSION
    ratings_updater_version: str = RATINGS_UPDATER_VERSION
    prior_equivalent_possessions: float = 6000.0
    minutes_updater_version: str = MINUTES_UPDATER_VERSION
    participation_prior_games: float = 4.0
    conditional_prior_games: float = 0.2
    conditional_lookback_games: int = 3
    conditional_clip_delta: float = 12.0
    rotation_size: int = 15
    role_cutoff_minutes: int = 44
    context_model_version: str = f"{CORE_MODEL_VERSION}:2025-26"
    matchup_url: str = DEFAULT_MATCHUP_URL

    def validate(self) -> None:
        if self.role_cutoff_minutes not in {36, 40, 44, 48}:
            raise ValueError("Role cutoff must be one of 36, 40, 44, or 48 minutes")
        InSeasonNailConfig(
            core_model_version=self.core_model_version,
            updater_version=self.ratings_updater_version,
            prior_equivalent_possessions=self.prior_equivalent_possessions,
        ).validate()
        self.minutes_config().validate()

    def minutes_config(self) -> InSeasonMinutesConfig:
        return InSeasonMinutesConfig(
            updater_version=self.minutes_updater_version,
            participation_prior_games=self.participation_prior_games,
            conditional_prior_games=self.conditional_prior_games,
            rotation_size=self.rotation_size,
            conditional_clip_delta=self.conditional_clip_delta,
            conditional_lookback_games=self.conditional_lookback_games,
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self))).hexdigest()


ContextProvider = Callable[
    [pd.DataFrame, bool, bool], tuple[np.ndarray, list[str]]
]


def replay_daily_model_chain(
    *,
    start_date: date,
    end_date: date,
    preseason_rankings_path: Path | str,
    preseason_minutes_path: Path | str,
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str = DEFAULT_MODEL_ROOT,
    config: DailyModelReplayConfig | None = None,
    code_revision: str | None = None,
    context_provider: ContextProvider | None = None,
) -> list[PublishedDailyModelArtifact]:
    """Replay verified dates in order and atomically advance only a clean tip.

    If an older data artifact is repaired, the longest still-valid model prefix
    is reused and every downstream date is rebuilt against the repaired chain.
    The old model pointer remains live until the entire requested replacement
    suffix has completed successfully.
    """

    config = config or DailyModelReplayConfig()
    if end_date < start_date:
        raise ValueError("Replay end date cannot precede its start date")
    config.validate()
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    rankings_path = Path(preseason_rankings_path).resolve()
    minutes_path = Path(preseason_minutes_path).resolve()
    rankings, minutes_priors, prior_sha = _load_frozen_priors(
        rankings_path, minutes_path, season=config.season
    )
    player_ids = tuple(sorted(rankings["player_id"].astype(int)))
    preseason = dict(
        zip(rankings["player_id"].astype(int), rankings["rapm"].astype(float), strict=True)
    )
    revision = code_revision or _git_revision()
    provenance = {
        "code_revision": revision,
        "model_version": config.model_version,
        "configuration_sha256": config.sha256,
        "prior_sha256": prior_sha,
    }

    data_chain = _read_data_chain(data_root)
    requested = [
        entry for entry in data_chain
        if start_date <= entry[0] <= end_date
    ]
    if not requested or requested[0][0] != start_date or requested[-1][0] != end_date:
        raise ValueError("Requested dates are not fully covered by the verified data chain")

    model_chain = _read_model_chain(output_root)
    compatible = _longest_compatible_prefix(
        data_chain=data_chain,
        model_chain=model_chain,
        provenance=provenance,
        through=end_date,
    )
    parent_bundle = compatible[-1][1] if compatible else None
    first_data_date = data_chain[0][0]
    next_date = compatible[-1][0] + timedelta(days=1) if compatible else first_data_date
    if next_date < start_date:
        next_date = start_date
    if parent_bundle is None and next_date != first_data_date:
        raise ValueError("A model replay without a compatible parent must begin at data genesis")

    if parent_bundle is None:
        rating_gram = sparse.csr_matrix((len(player_ids), len(player_ids)), dtype=float)
        rating_score = np.zeros(len(player_ids), dtype=float)
        rating_exposure = np.zeros(len(player_ids), dtype=float)
        minutes_state = initialize_minutes_state(
            minutes_priors, season=config.season, config=config.minutes_config()
        )
        prior_by_id = rankings.set_index("player_id")
        minutes_state["team"] = minutes_state["player_id"].map(
            prior_by_id["team"].astype(str)
        ).fillna(minutes_state["team"])
        trajectory = pd.DataFrame()
        team_by_player = rankings.set_index("player_id")["team"].astype(str).to_dict()
        last_team_date: dict[str, date] = {}
        contribution_lineage: list[str] = []
        included_game_ids: list[str] = []
    else:
        (
            rating_gram,
            rating_score,
            rating_exposure,
            minutes_state,
            trajectory,
            team_by_player,
            last_team_date,
            contribution_lineage,
            included_game_ids,
        ) = _load_parent_state(parent_bundle, expected_player_ids=player_ids)

    provider = context_provider or _http_context_provider(config.matchup_url)
    published: list[PublishedDailyModelArtifact] = []
    process_entries = [
        (game_date, bundle, manifest)
        for game_date, bundle, manifest in data_chain
        if max(next_date, start_date) <= game_date <= end_date
    ]
    try:
        for game_date, data_bundle, data_manifest in process_entries:
            with tempfile.TemporaryDirectory(prefix="daily-model-components-") as directory:
                component_root = Path(directory)
                result = _build_one_date(
                    game_date=game_date,
                    data_bundle=data_bundle,
                    data_manifest=data_manifest,
                    rankings=rankings,
                    preseason=preseason,
                    player_ids=player_ids,
                    rating_gram=rating_gram,
                    rating_score=rating_score,
                    rating_exposure=rating_exposure,
                    minutes_state=minutes_state,
                    trajectory=trajectory,
                    team_by_player=team_by_player,
                    last_team_date=last_team_date,
                    contribution_lineage=contribution_lineage,
                    included_game_ids=included_game_ids,
                    config=config,
                    context_provider=provider,
                    output_dir=component_root,
                )
                artifact = publish_daily_model_artifact(
                    root=output_root,
                    data_bundle=data_bundle,
                    input_data_uri=f"artifact://daily-data/{data_manifest['bundle_sha256']}",
                    parent_bundle=parent_bundle,
                    components=result["components"],
                    model_provenance=provenance,
                )
                published.append(artifact)
                parent_bundle = artifact.path
                rating_gram = result["rating_gram"]
                rating_score = result["rating_score"]
                rating_exposure = result["rating_exposure"]
                minutes_state = result["minutes_state"]
                trajectory = result["trajectory"]
                team_by_player = result["team_by_player"]
                last_team_date = result["last_team_date"]
                contribution_lineage = result["contribution_lineage"]
                included_game_ids = result["included_game_ids"]
    except Exception as error:
        failed_date = (
            process_entries[len(published)][0]
            if len(published) < len(process_entries)
            else end_date
        )
        report = _write_failure_report(output_root, failed_date, error)
        raise DailyModelReplayBlocked(
            f"Pacific date {failed_date.isoformat()} failed; report: {report}"
        ) from error

    final_tip = parent_bundle or (compatible[-1][1] if compatible else None)
    if final_tip is None:
        raise ValueError("Model replay produced no chain tip")
    _write_latest_pointer(output_root, final_tip)
    write_daily_model_chain_report(output_root)
    return published


def _build_one_date(
    *,
    game_date: date,
    data_bundle: Path,
    data_manifest: Mapping[str, Any],
    rankings: pd.DataFrame,
    preseason: Mapping[int, float],
    player_ids: tuple[int, ...],
    rating_gram: sparse.csr_matrix,
    rating_score: np.ndarray,
    rating_exposure: np.ndarray,
    minutes_state: pd.DataFrame,
    trajectory: pd.DataFrame,
    team_by_player: dict[int, str],
    last_team_date: dict[str, date],
    contribution_lineage: list[str],
    included_game_ids: list[str],
    config: DailyModelReplayConfig,
    context_provider: ContextProvider,
    output_dir: Path,
) -> dict[str, Any]:
    final_games = _read_json_component(data_bundle, data_manifest, "final_games")
    if not isinstance(final_games, list):
        raise ValueError("Final-games component must be a list")
    rosters_payload = _read_json_component(data_bundle, data_manifest, "rosters")
    injuries_payload = _read_json_component(data_bundle, data_manifest, "injuries")
    boxscores_payload = _read_json_component(data_bundle, data_manifest, "nba_boxscores")
    stints = pd.read_parquet(_component_path(data_bundle, data_manifest, "observed_stints"))
    observed_lineups = pd.read_parquet(
        _component_path(data_bundle, data_manifest, "observed_lineups")
    )
    rosters = {
        str(row["game_id"]): row
        for row in rosters_payload.get("games", [])
        if isinstance(row, dict)
    }
    boxscores = {
        str(row["game_id"]): row.get("payload")
        for row in boxscores_payload
        if isinstance(row, dict)
    }
    game_by_id = {str(row["game_id"]): row for row in final_games}
    expected_games = list(map(str, data_manifest["identity"]["game_ids"]))
    if sorted(game_by_id) != sorted(expected_games) or sorted(rosters) != sorted(expected_games):
        raise ValueError("Daily model inputs do not cover the exact verified slate")

    rating_ids = set(player_ids)
    minutes_ids = set(minutes_state["player_id"].astype(int))
    observed_ids = {
        int(player_id)
        for column in ("home_player_ids", "away_player_ids")
        for lineup in stints[column]
        for player_id in lineup
    }
    missing_ratings = sorted(observed_ids - rating_ids)
    missing_minutes = sorted(observed_ids - minutes_ids)
    if missing_ratings or missing_minutes:
        raise ValueError(
            "Player universe is incomplete: "
            f"ratings={missing_ratings}, minutes={missing_minutes}"
        )

    before_gram = rating_gram.copy()
    before_score = rating_score.copy()
    before_exposure = rating_exposure.copy()
    before_game_ids = list(included_game_ids)
    before_lineage = list(contribution_lineage)
    daily_contributions = []
    context_warnings: list[dict[str, Any]] = []
    b2b_by_game: dict[str, tuple[bool, bool]] = {}
    ordered_games = sorted(
        final_games, key=lambda row: (str(row["start_time"]), str(row["game_id"]))
    )
    for game in ordered_games:
        game_id = str(game["game_id"])
        home = str(game["home_team"])
        away = str(game["away_team"])
        home_b2b = last_team_date.get(home) == game_date - timedelta(days=1)
        away_b2b = last_team_date.get(away) == game_date - timedelta(days=1)
        b2b_by_game[game_id] = (home_b2b, away_b2b)
        game_stints = stints.loc[stints["game_id"].astype(str).eq(game_id)].copy()
        offsets, warnings = context_provider(game_stints, home_b2b, away_b2b)
        if warnings:
            context_warnings.append({"game_id": game_id, "warnings": warnings})
        start_time = pd.Timestamp(game["start_time"]).to_pydatetime()
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=UTC)
        generated_at = datetime.fromisoformat(
            str(data_manifest["generated_at"]).replace("Z", "+00:00")
        )
        finalized_at = max(start_time.astimezone(UTC), generated_at.astimezone(UTC))
        contribution = build_game_contribution(
            game_stints,
            season=config.season,
            game_id=game_id,
            game_time_utc=start_time,
            finalized_at=finalized_at,
            source_sha256=str(data_manifest["bundle_sha256"]),
            source_uri=f"artifact://daily-data/{data_manifest['bundle_sha256']}#{game_id}",
            core_model_version=config.core_model_version,
            context_model_version=config.context_model_version,
            player_ids=player_ids,
            preseason_nail=preseason,
            fixed_context_offset=offsets,
        )
        daily_contributions.append(contribution)
    for contribution in daily_contributions:
        rating_gram = rating_gram + contribution.xtwx
        rating_score = rating_score + contribution.xtwy
        rating_exposure = rating_exposure + contribution.on_court_possessions
        contribution_lineage.append(contribution.contribution_sha256)
        included_game_ids.append(contribution.game_id)

    for roster in rosters.values():
        for team in roster["teams"]:
            tricode = str(team["team"])
            for player in team["players"]:
                player_id = int(player["player_id"])
                team_by_player[player_id] = tricode
                mask = minutes_state["player_id"].astype(int).eq(player_id)
                if mask.any():
                    minutes_state.loc[mask, "team"] = tricode

    before_ratings = _current_rating_map(
        rankings=rankings,
        gram=before_gram,
        score=before_score,
        exposure=before_exposure,
        player_ids=player_ids,
        preseason=preseason,
        team_by_player=team_by_player,
        included_game_ids=before_game_ids,
        contribution_lineage=before_lineage,
        cutoff_time=datetime.combine(game_date, datetime.min.time(), tzinfo=UTC),
        config=config,
    )
    cutoff = datetime.combine(game_date + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    ratings_state = _solve_rating_state(
        rankings=rankings,
        gram=rating_gram,
        score=rating_score,
        exposure=rating_exposure,
        player_ids=player_ids,
        preseason=preseason,
        team_by_player=team_by_player,
        included_game_ids=included_game_ids,
        contribution_lineage=contribution_lineage,
        cutoff_time=cutoff,
        cutoff_game_id=str(ordered_games[-1]["game_id"]),
        config=config,
    )

    injury_report = _daily_injury_report(injuries_payload)
    observations, injury_rows_by_game, injury_warnings = _minutes_observations(
        game_date=game_date,
        games=game_by_id,
        rosters=rosters,
        stints=stints,
        injury_report=injury_report,
        minutes_state=minutes_state,
        role_cutoff_minutes=config.role_cutoff_minutes,
    )
    observation_sha = hashlib.sha256(
        observations.sort_values(["game_id", "team", "player_id"], kind="stable")
        .to_json(orient="records", double_precision=12)
        .encode()
    ).hexdigest()
    minutes_state = advance_minutes_state(
        minutes_state,
        observations,
        config=config.minutes_config(),
        contribution_sha256=observation_sha,
    )

    trajectory_rows = _trajectory_rows(
        game_date=game_date,
        games=game_by_id,
        rosters=rosters,
        boxscores=boxscores,
        observations=observations,
        before=before_ratings,
        after=ratings_state,
    )
    trajectory = pd.concat([trajectory, trajectory_rows], ignore_index=True)
    if trajectory.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Rating trajectory duplicates a player-game")

    allocations, game_projections = _descriptive_game_projections(
        games=ordered_games,
        rosters=rosters,
        injury_rows_by_game=injury_rows_by_game,
        minutes_state=minutes_state,
        ratings_state=ratings_state,
        b2b_by_game=b2b_by_game,
        config=config,
    )
    lineup_values = _lineup_values(observed_lineups, ratings_state)

    for game in ordered_games:
        last_team_date[str(game["home_team"])] = game_date
        last_team_date[str(game["away_team"])] = game_date

    diagnostics = {
        "status": "clean",
        "game_date_pacific": game_date.isoformat(),
        "input_data_sha256": data_manifest["bundle_sha256"],
        "daily_game_count": len(expected_games),
        "cumulative_game_count": len(set(included_game_ids)),
        "player_count": len(ratings_state),
        "observed_player_count": len(observed_ids),
        "missing_rating_player_ids": [],
        "missing_minutes_player_ids": [],
        "non_finite_rating_count": int(
            (~np.isfinite(ratings_state["updated_nail"].to_numpy(dtype=float))).sum()
        ),
        "allocation_team_game_count": int(
            allocations[["game_id", "team"]].drop_duplicates().shape[0]
        ),
        "allocation_max_absolute_error": float(
            allocations.groupby(["game_id", "team"])["final_normalized_minutes"]
            .sum().sub(240.0).abs().max()
        ),
        "allocation_max_player_minutes": float(allocations["final_normalized_minutes"].max()),
        "unknown_minutes_observations": int(
            observations["outcome_classification"].eq(UNKNOWN).sum()
        ),
        "context_warnings": context_warnings,
        "injury_identity_warnings": injury_warnings,
        "blake_wesley_prior_present": 1631104 in set(player_ids),
        "excluded_game_count": 0,
        "excluded_games": [],
        "projection_kind": "post_date_state_descriptive_not_pregame",
    }
    if diagnostics["non_finite_rating_count"]:
        raise ValueError("Daily rating state contains non-finite values")
    if diagnostics["allocation_max_absolute_error"] > 1e-8:
        raise ValueError("A game-team minutes allocation does not sum to 240")
    if diagnostics["allocation_max_player_minutes"] > 48.0 + 1e-8:
        raise ValueError("A player allocation exceeds 48 regulation minutes")

    components = _write_model_components(
        output_dir,
        ratings_state=ratings_state,
        minutes_state=minutes_state,
        trajectory=trajectory,
        lineup_values=lineup_values,
        game_projections=game_projections,
        diagnostics=diagnostics,
        rating_gram=rating_gram,
        rating_score=rating_score,
        rating_exposure=rating_exposure,
        observations=observations,
        allocations=allocations,
        team_by_player=team_by_player,
        last_team_date=last_team_date,
        contribution_lineage=contribution_lineage,
        included_game_ids=included_game_ids,
    )
    return {
        "components": components,
        "rating_gram": rating_gram,
        "rating_score": rating_score,
        "rating_exposure": rating_exposure,
        "minutes_state": minutes_state,
        "trajectory": trajectory,
        "team_by_player": team_by_player,
        "last_team_date": last_team_date,
        "contribution_lineage": contribution_lineage,
        "included_game_ids": included_game_ids,
    }


def _solve_rating_state(
    *,
    rankings: pd.DataFrame,
    gram: sparse.csr_matrix,
    score: np.ndarray,
    exposure: np.ndarray,
    player_ids: tuple[int, ...],
    preseason: Mapping[int, float],
    team_by_player: Mapping[int, str],
    included_game_ids: Sequence[str],
    contribution_lineage: Sequence[str],
    cutoff_time: datetime,
    cutoff_game_id: str | None,
    config: DailyModelReplayConfig,
) -> pd.DataFrame:
    prior = np.asarray([float(preseason[player_id]) for player_id in player_ids])
    prior_precision = np.full(len(player_ids), config.prior_equivalent_possessions)
    precision = gram + sparse.diags(prior_precision, format="csr")
    adjustment = (
        np.zeros(len(player_ids), dtype=float)
        if not included_game_ids
        else np.asarray(spsolve(precision, score), dtype=float)
    )
    if not np.isfinite(adjustment).all():
        raise ValueError("In-season rating solve produced non-finite values")
    information = np.asarray(gram.diagonal(), dtype=float)
    evidence_weight = information / (information + prior_precision)
    posterior_variance = np.diag(np.linalg.inv(precision.toarray()))
    name_by_id = rankings.set_index("player_id")["player_name"].astype(str).to_dict()
    state = pd.DataFrame(
        {
            "schema_version": 1,
            "season": config.season,
            "core_model_version": config.core_model_version,
            "state_updater_version": config.ratings_updater_version,
            "player_id": player_ids,
            "player_name": [name_by_id[player_id] for player_id in player_ids],
            "team_at_cutoff": [team_by_player.get(player_id) for player_id in player_ids],
            "cutoff_time": cutoff_time,
            "cutoff_game_id": cutoff_game_id,
            "frozen_preseason_nail": prior,
            "current_season_residual": adjustment,
            "updated_nail": prior + adjustment,
            "raw_on_court_possessions": exposure,
            "effective_information": information,
            "prior_equivalent_possessions": prior_precision,
            "prior_weight": 1.0 - evidence_weight,
            "evidence_weight": evidence_weight,
            "posterior_standard_error": np.sqrt(np.maximum(posterior_variance, 0.0)),
            "included_game_ids_json": json.dumps(tuple(included_game_ids)),
            "contribution_lineage_json": json.dumps(tuple(contribution_lineage)),
        }
    )
    validate_player_state(state)
    return state


def _current_rating_map(**kwargs: Any) -> pd.DataFrame:
    return _solve_rating_state(cutoff_game_id=None, config=kwargs.pop("config"), **kwargs)


def _minutes_observations(
    *,
    game_date: date,
    games: Mapping[str, Mapping[str, Any]],
    rosters: Mapping[str, Mapping[str, Any]],
    stints: pd.DataFrame,
    injury_report: Mapping[str, Any],
    minutes_state: pd.DataFrame,
    role_cutoff_minutes: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], list[str]]:
    state = minutes_state.copy()
    state["player_id"] = state["player_id"].astype(int)
    identity: dict[tuple[str, str], list[int]] = {}
    for row in state.itertuples(index=False):
        identity.setdefault((str(row.team), _normalize_name(row.player_name)), []).append(
            int(row.player_id)
        )
    rows: list[dict[str, object]] = []
    injury_by_game: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for game_id, game in sorted(games.items()):
        matchup = f"{game['away_team']}@{game['home_team']}"
        player_reports = [
            row for row in injury_report.get("players", [])
            if str(row.get("game_date")) == game_date.isoformat()
            and str(row.get("matchup")) == matchup
        ]
        game_injuries: dict[str, list[dict[str, object]]] = {
            str(game["home_team"]): [], str(game["away_team"]): []
        }
        report_key = f"{game_date.isoformat()}|{matchup}"
        team_reports = injury_report.get("game_team_reports", {}).get(report_key, {})
        report_status = {
            team: str(team_reports.get(team, {}).get("status", "not_yet_submitted"))
            for team in game_injuries
        }
        mapped_report: dict[tuple[str, str], dict[str, object]] = {}
        for report in player_reports:
            team = str(report.get("team", ""))
            key = (team, _normalize_name(report.get("player_name")))
            candidates = identity.get(key, [])
            if len(candidates) == 1:
                normalized = dict(report)
                normalized["player_id"] = candidates[0]
                mapped_report[key] = normalized
                game_injuries.setdefault(team, []).append(normalized)
            else:
                warnings.append(
                    f"{game_id}/{team}/{report.get('player_name')}: "
                    + ("unmapped" if not candidates else "ambiguous")
                )
        injury_by_game[game_id] = {
            "teams": game_injuries,
            "report_status": report_status,
        }
        game_stints = stints.loc[stints["game_id"].astype(str).eq(game_id)]
        roles = competitive_role_minutes(
            game_stints, boundary_seconds=float(role_cutoff_minutes * 60)
        )
        roster = rosters[game_id]
        seen: set[int] = set()
        for team_row in roster["teams"]:
            team = str(team_row["team"])
            team_players = list(team_row["players"])
            total = sum(float(player.get("minutes", 0.0)) for player in team_players)
            if total <= 0:
                raise ValueError(f"Official box score has no team minutes for {game_id}/{team}")
            equivalent = regulation_equivalent_minutes(
                pd.Series([float(player.get("minutes", 0.0)) for player in team_players]),
                team_total_minutes=total,
            )
            for player, regulated in zip(team_players, equivalent, strict=True):
                player_id = int(player["player_id"])
                seen.add(player_id)
                report = mapped_report.get((team, _normalize_name(player["player_name"])), {})
                minutes = float(player.get("minutes", 0.0))
                classification = _classification(
                    minutes=minutes,
                    roster_status=player.get("status"),
                    injury_status=report.get("status"),
                    injury_reason=report.get("reason"),
                )
                role = float(roles.get(player_id, 0.0))
                if minutes > 0 and player_id not in roles:
                    post_cutoff = 48.0 - role_cutoff_minutes
                    if minutes > post_cutoff + 0.25:
                        raise ValueError(
                            f"44-minute stints omit played player {player_id} in {game_id}"
                        )
                rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date.isoformat(),
                        "team": team,
                        "player_id": player_id,
                        "player_name": str(player["player_name"]),
                        "outcome_classification": classification,
                        "played": minutes > 0,
                        "actual_minutes": minutes,
                        "regulation_equivalent_minutes": float(regulated),
                        "early_game_role_minutes": role,
                        "conditional_observation_weight": 1.0,
                        "injury_status": report.get("status"),
                        "injury_reason": report.get("reason"),
                        "roster_status": player.get("status"),
                    }
                )
        for team, injury_rows in game_injuries.items():
            for report in injury_rows:
                player_id = int(report["player_id"])
                if player_id in seen:
                    continue
                classification = _classification(
                    minutes=0.0,
                    roster_status="",
                    injury_status=report.get("status"),
                    injury_reason=report.get("reason"),
                )
                rows.append(
                    {
                        "game_id": game_id,
                        "game_date": game_date.isoformat(),
                        "team": team,
                        "player_id": player_id,
                        "player_name": str(report.get("player_name")),
                        "outcome_classification": classification,
                        "played": False,
                        "actual_minutes": 0.0,
                        "regulation_equivalent_minutes": 0.0,
                        "early_game_role_minutes": 0.0,
                        "conditional_observation_weight": 1.0,
                        "injury_status": report.get("status"),
                        "injury_reason": report.get("reason"),
                        "roster_status": None,
                    }
                )
    output = pd.DataFrame(rows).sort_values(
        ["game_id", "team", "player_id"], kind="stable"
    ).reset_index(drop=True)
    if output.duplicated(["game_id", "player_id"]).any():
        raise ValueError("Minutes observations duplicate a player-game")
    return output, injury_by_game, warnings


def _classification(
    *,
    minutes: float,
    roster_status: object,
    injury_status: object,
    injury_reason: object,
) -> str:
    if minutes > 0:
        return PLAYED
    status = str(injury_status or "").strip().upper()
    text = str(injury_reason or "").upper()
    if any(token in text for token in G_LEAGUE_TOKENS):
        return G_LEAGUE_ASSIGNMENT
    if any(token in text for token in INACTIVE_TOKENS):
        return INACTIVE
    if status in INJURY_STATUSES:
        return MEDICAL_ABSENCE
    if status == "AVAILABLE" or str(roster_status or "").upper() == "ACTIVE":
        return COACH_DNP
    if str(roster_status or "").upper() == "INACTIVE":
        return INACTIVE
    return UNKNOWN


def _descriptive_game_projections(
    *,
    games: Sequence[Mapping[str, Any]],
    rosters: Mapping[str, Mapping[str, Any]],
    injury_rows_by_game: Mapping[str, Mapping[str, Any]],
    minutes_state: pd.DataFrame,
    ratings_state: pd.DataFrame,
    b2b_by_game: Mapping[str, tuple[bool, bool]],
    config: DailyModelReplayConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rating = ratings_state.set_index("player_id")["updated_nail"].astype(float).to_dict()
    allocation_frames: list[pd.DataFrame] = []
    projections: list[dict[str, object]] = []
    for game in games:
        game_id = str(game["game_id"])
        roster = rosters[game_id]
        roster_by_team = {
            str(row["team"]): {int(player["player_id"]) for player in row["players"]}
            for row in roster["teams"]
        }
        side_strength: dict[str, float] = {}
        for team in (str(game["home_team"]), str(game["away_team"])):
            injury_game = injury_rows_by_game.get(game_id, {})
            team_injuries = injury_game.get("teams", {}).get(team, [])
            report_status = str(
                injury_game.get("report_status", {}).get(team, "not_yet_submitted")
            )
            allocation = forecast_team_minutes(
                minutes_state,
                team=team,
                game_id=game_id,
                forecast_cutoff="post-date descriptive state",
                injury_rows=team_injuries,
                report_status=report_status,
                config=config.minutes_config(),
                roster_player_ids=roster_by_team[team],
            )
            missing = sorted(set(allocation["player_id"].astype(int)) - set(rating))
            if missing:
                raise ValueError(f"Allocation players lack ratings: {missing}")
            allocation["rating"] = allocation["player_id"].astype(int).map(rating)
            allocation["weighted_rating"] = (
                allocation["final_normalized_minutes"] / 48.0 * allocation["rating"]
            )
            side_strength[team] = float(allocation["weighted_rating"].sum())
            allocation_frames.append(allocation)
        home = str(game["home_team"])
        away = str(game["away_team"])
        margin = side_strength[home] - side_strength[away]
        home_b2b, away_b2b = b2b_by_game[game_id]
        projections.append(
            {
                "game_id": game_id,
                "game_date_pacific": game["game_date_pacific"],
                "home_team": home,
                "away_team": away,
                "home_team_strength": side_strength[home],
                "away_team_strength": side_strength[away],
                "neutral_margin": margin,
                "home_win_probability": projected_win_pct(margin),
                "home_back_to_back": home_b2b,
                "away_back_to_back": away_b2b,
                "actual_home_margin": float(game["actual_home_margin"]),
                "projection_kind": "post_date_state_descriptive_not_pregame",
                "eligible_for_forward_evaluation": False,
            }
        )
    return pd.concat(allocation_frames, ignore_index=True), pd.DataFrame(projections)
def _lineup_values(lineups: pd.DataFrame, ratings_state: pd.DataFrame) -> pd.DataFrame:
    rating = ratings_state.set_index("player_id")["updated_nail"].astype(float).to_dict()
    output = lineups.copy()
    output["player_ids"] = output["player_ids_json"].map(json.loads)
    missing = sorted(
        {
            int(player_id)
            for lineup in output["player_ids"]
            for player_id in lineup
            if int(player_id) not in rating
        }
    )
    if missing:
        raise ValueError(f"Observed lineups lack player ratings: {missing}")
    output["additive_lineup_value"] = output["player_ids"].map(
        lambda lineup: sum(rating[int(player_id)] for player_id in lineup)
    )
    output["player_count"] = output["player_ids"].map(len)
    if not output["player_count"].eq(5).all():
        raise ValueError("Observed lineup component contains a non-five-player unit")
    return output.drop(columns="player_ids")


def _trajectory_rows(
    *,
    game_date: date,
    games: Mapping[str, Mapping[str, Any]],
    rosters: Mapping[str, Mapping[str, Any]],
    boxscores: Mapping[str, Any],
    observations: pd.DataFrame,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> pd.DataFrame:
    before_map = before.set_index("player_id")["updated_nail"].astype(float).to_dict()
    after_map = after.set_index("player_id")["updated_nail"].astype(float).to_dict()
    observation_map = {
        (str(row.game_id), int(row.player_id)): row
        for row in observations.itertuples(index=False)
    }
    plus_minus = _boxscore_plus_minus(boxscores)
    rows: list[dict[str, object]] = []
    for game_id, roster in rosters.items():
        game = games[game_id]
        for team_row in roster["teams"]:
            team = str(team_row["team"])
            opponent = (
                str(game["away_team"]) if team == str(game["home_team"])
                else str(game["home_team"])
            )
            for player in team_row["players"]:
                player_id = int(player["player_id"])
                if player_id not in before_map or player_id not in after_map:
                    raise ValueError(f"Trajectory player lacks rating state: {player_id}")
                observation = observation_map.get((game_id, player_id))
                rows.append(
                    {
                        "season": str(game["season"]),
                        "game_date_pacific": game_date.isoformat(),
                        "game_id": game_id,
                        "game_time_utc": game["start_time"],
                        "player_id": player_id,
                        "player_name": str(player["player_name"]),
                        "team": team,
                        "opponent": opponent,
                        "played": bool(player.get("played")),
                        "actual_minutes": float(player.get("minutes", 0.0)),
                        "raw_plus_minus": plus_minus.get((game_id, player_id)),
                        "outcome_classification": (
                            str(observation.outcome_classification)
                            if observation is not None else UNKNOWN
                        ),
                        "injury_status": getattr(observation, "injury_status", None),
                        "injury_reason": getattr(observation, "injury_reason", None),
                        "rating_before": before_map[player_id],
                        "rating_after": after_map[player_id],
                        "rating_delta": after_map[player_id] - before_map[player_id],
                    }
                )
    return pd.DataFrame(rows)


def _boxscore_plus_minus(
    boxscores: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], float | None]:
    output: dict[tuple[str, int], float | None] = {}
    for game_id, payload in boxscores.items():
        game = payload.get("game", {}) if isinstance(payload, dict) else {}
        for side in ("homeTeam", "awayTeam"):
            for player in game.get(side, {}).get("players", []):
                value = player.get("statistics", {}).get("plusMinusPoints")
                output[(game_id, int(player["personId"]))] = (
                    float(value) if value not in (None, "") else None
                )
    return output


def _write_model_components(
    root: Path,
    *,
    ratings_state: pd.DataFrame,
    minutes_state: pd.DataFrame,
    trajectory: pd.DataFrame,
    lineup_values: pd.DataFrame,
    game_projections: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    rating_gram: sparse.csr_matrix,
    rating_score: np.ndarray,
    rating_exposure: np.ndarray,
    observations: pd.DataFrame,
    allocations: pd.DataFrame,
    team_by_player: Mapping[int, str],
    last_team_date: Mapping[str, date],
    contribution_lineage: Sequence[str],
    included_game_ids: Sequence[str],
) -> dict[str, Path]:
    paths = {
        "player_ratings": root / "player_ratings.parquet",
        "minutes_state": root / "minutes_state.parquet",
        "rating_trajectory": root / "rating_trajectory.parquet",
        "lineup_values": root / "lineup_values.parquet",
        "game_projections": root / "game_projections.parquet",
        "evaluation_metrics": root / "evaluation_metrics.json",
        "rating_gram": root / "rating_gram.npz",
        "rating_vectors": root / "rating_vectors.npz",
        "minutes_observations": root / "minutes_observations.parquet",
        "minute_allocations": root / "minute_allocations.parquet",
        "replay_state": root / "replay_state.json",
    }
    ratings_state.to_parquet(paths["player_ratings"], index=False)
    minutes_state.to_parquet(paths["minutes_state"], index=False)
    trajectory.to_parquet(paths["rating_trajectory"], index=False)
    lineup_values.to_parquet(paths["lineup_values"], index=False)
    game_projections.to_parquet(paths["game_projections"], index=False)
    paths["evaluation_metrics"].write_text(
        json.dumps(dict(diagnostics), indent=2, sort_keys=True) + "\n"
    )
    sparse.save_npz(paths["rating_gram"], rating_gram, compressed=True)
    np.savez_compressed(
        paths["rating_vectors"], score=rating_score, exposure=rating_exposure
    )
    observations.to_parquet(paths["minutes_observations"], index=False)
    allocations.to_parquet(paths["minute_allocations"], index=False)
    paths["replay_state"].write_text(
        json.dumps(
            {
                "team_by_player": {str(key): value for key, value in team_by_player.items()},
                "last_team_date": {
                    key: value.isoformat() for key, value in last_team_date.items()
                },
                "contribution_lineage": list(contribution_lineage),
                "included_game_ids": list(included_game_ids),
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )
    return paths


def _load_parent_state(
    bundle: Path, *, expected_player_ids: tuple[int, ...]
) -> tuple[
    sparse.csr_matrix,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, str],
    dict[str, date],
    list[str],
    list[str],
]:
    manifest = verify_daily_model_artifact(bundle)
    gram = sparse.load_npz(_component_path(bundle, manifest, "rating_gram"))
    with np.load(_component_path(bundle, manifest, "rating_vectors")) as vectors:
        score = vectors["score"].copy()
        exposure = vectors["exposure"].copy()
    if gram.shape != (len(expected_player_ids), len(expected_player_ids)):
        raise ValueError("Parent rating vocabulary differs from the frozen prior")
    minutes = pd.read_parquet(_component_path(bundle, manifest, "minutes_state"))
    trajectory = pd.read_parquet(_component_path(bundle, manifest, "rating_trajectory"))
    state = _read_json(_component_path(bundle, manifest, "replay_state"))
    return (
        gram,
        score,
        exposure,
        minutes,
        trajectory,
        {int(key): str(value) for key, value in state["team_by_player"].items()},
        {key: date.fromisoformat(value) for key, value in state["last_team_date"].items()},
        list(map(str, state["contribution_lineage"])),
        list(map(str, state["included_game_ids"])),
    )


def _load_frozen_priors(
    rankings_path: Path, minutes_path: Path, *, season: str
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rankings = pd.read_parquet(rankings_path).copy()
    required = {"season", "player_id", "player_name", "team", "rapm"}
    missing = sorted(required - set(rankings))
    if missing or rankings.empty or rankings["player_id"].duplicated().any():
        raise ValueError(f"Frozen ratings prior is invalid; missing={missing}")
    rankings = rankings.loc[rankings["season"].astype(str).eq(season)].copy()
    rankings["player_id"] = rankings["player_id"].astype(int)
    rankings["rapm"] = pd.to_numeric(rankings["rapm"], errors="raise").astype(float)
    if not np.isfinite(rankings["rapm"]).all():
        raise ValueError("Frozen ratings prior contains non-finite values")
    payload = _read_json(minutes_path)
    minutes_payload = payload.get("minutes", payload)
    players = minutes_payload.get("players") if isinstance(minutes_payload, dict) else None
    if not isinstance(players, list):
        raise ValueError("Frozen minutes prior has no player catalog")
    minutes = pd.DataFrame(players)
    required_minutes = {
        "player_id", "player_name", "team", "availability_probability",
        "conditional_minutes_per_game", "baseline_minutes_per_game",
    }
    missing_minutes = sorted(required_minutes - set(minutes))
    if missing_minutes or minutes["player_id"].duplicated().any():
        raise ValueError(f"Frozen minutes prior is invalid; missing={missing_minutes}")
    minutes["player_id"] = minutes["player_id"].astype(int)
    ranking_metadata = rankings_path.with_suffix(".json")
    prior_inventory = {
        "rankings_sha256": file_sha256(rankings_path),
        "rankings_metadata_sha256": (
            file_sha256(ranking_metadata) if ranking_metadata.is_file() else None
        ),
        "minutes_sha256": file_sha256(minutes_path),
    }
    prior_sha = hashlib.sha256(canonical_json(prior_inventory)).hexdigest()
    return (
        rankings.sort_values("player_id", kind="stable").reset_index(drop=True),
        minutes.sort_values("player_id", kind="stable").reset_index(drop=True),
        prior_sha,
    )


def _http_context_provider(matchup_url: str) -> ContextProvider:
    cache: dict[tuple[object, ...], tuple[float, str | None]] = {}

    def provide(
        stints: pd.DataFrame, home_back_to_back: bool, away_back_to_back: bool
    ) -> tuple[np.ndarray, list[str]]:
        return _context_offsets(
            stints,
            matchup_url=matchup_url,
            home_back_to_back=home_back_to_back,
            away_back_to_back=away_back_to_back,
            cache=cache,
        )

    return provide


def _read_data_chain(root: Path) -> list[tuple[date, Path, dict[str, Any]]]:
    pointer = _read_json(root / "latest.json")
    bundle = (root / str(pointer["bundle_path"])).resolve()
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    while True:
        manifest = verify_daily_data_artifact(bundle)
        identity = manifest["identity"]
        game_date = date.fromisoformat(identity["game_date_pacific"])
        rows.append((game_date, bundle, manifest))
        parent_sha = identity["parent_data_sha256"]
        if parent_sha is None:
            break
        prior_date = game_date - timedelta(days=1)
        bundle = (
            root / f"season={identity['season']}" / f"game_date_pacific={prior_date.isoformat()}"
            / f"data_sha256={parent_sha}"
        )
    rows.reverse()
    return rows


def _read_model_chain(root: Path) -> list[tuple[date, Path, dict[str, Any]]]:
    pointer_path = root / "latest.json"
    if not pointer_path.is_file():
        return []
    pointer = _read_json(pointer_path)
    bundle = (root / str(pointer["bundle_path"])).resolve()
    rows: list[tuple[date, Path, dict[str, Any]]] = []
    while True:
        manifest = verify_daily_model_artifact(bundle)
        identity = manifest["identity"]
        game_date = date.fromisoformat(identity["game_date_pacific"])
        rows.append((game_date, bundle, manifest))
        parent_sha = identity["parent_model_sha256"]
        if parent_sha is None:
            break
        prior_date = game_date - timedelta(days=1)
        bundle = (
            root / f"season={identity['season']}" / f"game_date_pacific={prior_date.isoformat()}"
            / f"model_sha256={parent_sha}"
        )
    rows.reverse()
    return rows


def _longest_compatible_prefix(
    *,
    data_chain: Sequence[tuple[date, Path, Mapping[str, Any]]],
    model_chain: Sequence[tuple[date, Path, Mapping[str, Any]]],
    provenance: Mapping[str, str],
    through: date,
) -> list[tuple[date, Path, Mapping[str, Any]]]:
    output: list[tuple[date, Path, Mapping[str, Any]]] = []
    for data_row, model_row in zip(data_chain, model_chain, strict=False):
        data_date, _data_path, data_manifest = data_row
        model_date, model_path, model_manifest = model_row
        identity = model_manifest["identity"]
        if data_date > through:
            break
        if (
            data_date != model_date
            or identity["input_data_sha256"] != data_manifest["bundle_sha256"]
            or identity["model_provenance"] != dict(provenance)
        ):
            break
        output.append((model_date, model_path, model_manifest))
    return output


def _component_path(
    bundle: Path, manifest: Mapping[str, Any], name: str
) -> Path:
    entry = manifest["identity"]["components"].get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"Artifact component is missing: {name}")
    return bundle / str(entry["path"])


def _read_json_component(
    bundle: Path, manifest: Mapping[str, Any], name: str
) -> Any:
    return json.loads(_component_path(bundle, manifest, name).read_text())


def _daily_injury_report(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("Daily data artifact must contain exactly one NBA injury report")
    report = payload[0].get("payload") if isinstance(payload[0], dict) else None
    if not isinstance(report, dict):
        raise ValueError("NBA injury report payload is missing")
    return report


def _normalize_name(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    text = re.sub(r"\s+(?:Jr\.?|Sr\.?|II|III|IV|V)$", "", text, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", text.casefold())


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _write_latest_pointer(root: Path, bundle: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = verify_daily_model_artifact(bundle)
    relative = bundle.resolve().relative_to(root.resolve())
    pointer = {
        "schema_version": 1,
        "artifact_kind": "daily_model",
        "season": manifest["identity"]["season"],
        "game_date_pacific": manifest["identity"]["game_date_pacific"],
        "bundle_sha256": manifest["bundle_sha256"],
        "bundle_path": relative.as_posix(),
        "promoted_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    temporary = root / ".latest.json.tmp"
    temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
    temporary.replace(root / "latest.json")
    history = root / "pointer_history" / f"{manifest['bundle_sha256']}.json"
    history.parent.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        history.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")


def _write_failure_report(root: Path, game_date: date, error: Exception) -> Path:
    failure = root / "failures" / f"game_date_pacific={game_date.isoformat()}"
    failure.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_date_pacific": game_date.isoformat(),
        "error": f"{type(error).__name__}: {error}",
    }
    (failure / "error.json").write_text(json.dumps(payload, indent=2) + "\n")
    report = failure / "report.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>Daily model replay blocked</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1000px}.bad{color:#a11}</style>"
        f"<h1 class='bad'>Model replay blocked — {html.escape(game_date.isoformat())}</h1>"
        "<p>No model pointer was advanced.</p>"
        f"<pre>{html.escape(payload['error'])}</pre>",
        encoding="utf-8",
    )
    return report


def write_daily_model_chain_report(root: Path | str) -> Path:
    chain_root = Path(root).resolve()
    chain = _read_model_chain(chain_root)
    rows: list[str] = []
    for game_date, bundle, manifest in chain:
        diagnostics = _read_json(_component_path(bundle, manifest, "evaluation_metrics"))
        rows.append(
            "<tr>"
            f"<td>{game_date.isoformat()}</td>"
            f"<td>{len(manifest['identity']['daily_game_ids'])}</td>"
            f"<td>{diagnostics['cumulative_game_count']}</td>"
            f"<td>{diagnostics['player_count']}</td>"
            f"<td>{diagnostics['unknown_minutes_observations']}</td>"
            f"<td><code>{manifest['bundle_sha256'][:12]}</code></td>"
            "<td class='ok'>CLEAN</td></tr>"
        )
    tip = chain[-1][2]
    report = chain_root / "report.html"
    report.write_text(
        "<!doctype html><meta charset='utf-8'><title>Daily NBA model chain</title>"
        "<style>body{font:16px system-ui;margin:2rem;max-width:1100px}"
        "table{border-collapse:collapse;width:100%}td,th{padding:.55rem;"
        "border-bottom:1px solid #ddd;text-align:left}.ok{color:#176b3a;font-weight:700}</style>"
        f"<h1>Official NBA daily model chain</h1><p class='ok'>CLEAN THROUGH "
        f"{html.escape(tip['identity']['game_date_pacific'])}</p>"
        f"<p>{len(chain)} Pacific dates · {tip['identity']['cumulative_game_count']} games · "
        "runtime ESPN dependencies: 0</p>"
        "<p>Pregame evaluation is intentionally absent from this state-chain report; "
        "the projection component is explicitly post-date descriptive.</p>"
        "<table><thead><tr><th>Date</th><th>Daily games</th><th>Cumulative</th>"
        "<th>Players</th><th>Unknown minute rows</th><th>Model SHA</th><th>Status</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>",
        encoding="utf-8",
    )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--preseason-rankings-path", type=Path, required=True)
    parser.add_argument("--preseason-minutes-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--matchup-url", default=DEFAULT_MATCHUP_URL)
    parser.add_argument("--code-revision")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = DailyModelReplayConfig(matchup_url=args.matchup_url)
    artifacts = replay_daily_model_chain(
        start_date=args.start_date,
        end_date=args.end_date,
        preseason_rankings_path=args.preseason_rankings_path,
        preseason_minutes_path=args.preseason_minutes_path,
        data_root=args.data_root,
        output_root=args.output_root,
        config=config,
        code_revision=args.code_revision,
    )
    for artifact in artifacts:
        print(artifact.path)


if __name__ == "__main__":
    main()
