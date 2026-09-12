"""Forward-safe NAIL tilt for the promoted preseason roster-minute squash.

This experiment nests the live Forward Availability + Forward Conditional
Minutes forecast exactly.  At ``nail_beta == 0`` it retains each team's same
top-15 raw expected-minute weights and normalization.  Nonzero beta only
reweights those forecast weights using the NAIL rating that was available
before the target season began.

The target season is used solely for the full-regular-season allocation label.
The opening roster, target biographies, availability state, conditional
minutes state, and NAIL feature profile are all constrained to preseason
information.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
    predict_availability_roster,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    ColdStartConditionalMinutesConfig,
    attach_cold_start_biographies,
    attach_expected_total_minutes,
    normalize_opening_roster_minutes,
    predict_conditional_minutes_roster,
)
from nba_lineup_model.rotation.l0_opening_minute_share_persistence import (
    read_preseason_roster_candidates,
)
from nba_lineup_model.rotation.l1_minute_share_persistence import (
    DEFAULT_ARTIFACTS_DIR,
    read_regular_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_minute_share_persistence import (
    _assert_distribution,
    _first_n_team_game_allocations,
    _validate_game_minutes,
    summarize_l20_metrics,
)
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_MODEL_RUN_DIR,
    DEFAULT_PANEL_PATH,
    build_preseason_nail_forecast_ratings,
)

MODEL_NAME = "forward_nail_roster_squash"
MODEL_VERSION = "v0.1"
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_NAIL_BETA_GRID = (-0.40, -0.20, -0.10, 0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
DEFAULT_INITIAL_ROTATION_SIZE = 15
DEFAULT_CATALOG_PATH = Path("data/catalog/games.parquet")
_BIOGRAPHY_COLUMNS = (
    "player_id",
    "age",
    "draft_year",
    "draft_number",
    "is_undrafted",
    "listed_position",
    "is_rookie",
    "season_start_year",
)


@dataclass(frozen=True)
class ForwardNailRosterSquashInputs:
    """Preseason inputs plus a held-out, post-season allocation label."""

    season: str
    opening_roster: pd.DataFrame
    static_roster_bios: pd.DataFrame
    preseason_nail: pd.Series
    target_game_minutes: pd.DataFrame


@dataclass(frozen=True)
class ForwardNailRosterSquashRun:
    """Immutable artifact location for a completed candidate replay."""

    run_dir: Path
    run_id: str


@dataclass(frozen=True)
class _SeasonAllocationTarget:
    """Observed full-season team allocation held out from a preseason forecast."""

    team_id: int
    team: str
    first_game_id: str
    last_game_id: str
    target_game_count: int
    player_shares: dict[int, float]


def build_static_opening_roster_bios(
    season: str,
    opening_roster: pd.DataFrame,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> pd.DataFrame:
    """Attach only target-static biography fields to an opening roster.

    The panel contains completed outcomes, but this function selects only age,
    position, rookie status, and draft fields.  A missing target row falls back
    to the player's latest earlier biography with age advanced by calendar
    seasons.  Players with no historical panel row use a neutral static
    fallback; target-season minutes, games, and box-score fields are never
    read.
    """

    required = {"team_id", "team", "player_id", "player_name"}
    missing = sorted(required - set(opening_roster))
    if missing:
        raise ValueError("Opening roster lacks required columns: " + ", ".join(missing))
    target_year = _season_year(season)
    roster = opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]].copy()
    roster["team_id"] = pd.to_numeric(roster["team_id"], errors="raise").astype(int)
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype(int)
    if roster.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Opening roster contains duplicate player-team rows")

    static_panel = panel.loc[
        :, ["season", *[column for column in _BIOGRAPHY_COLUMNS if column in panel]]
    ].copy()
    static_panel["player_id"] = pd.to_numeric(static_panel["player_id"], errors="raise").astype(
        int
    )
    target = static_panel.loc[
        static_panel["season"].astype(str).eq(season),
        [column for column in _BIOGRAPHY_COLUMNS if column in static_panel],
    ].copy()
    target = target.loc[target["player_id"].isin(roster["player_id"])].drop_duplicates(
        "player_id", keep="last"
    )
    target["static_bio_source"] = "target_static_biography"

    missing_ids = sorted(set(roster["player_id"]) - set(target["player_id"]))
    historical = pd.DataFrame(columns=target.columns)
    if missing_ids:
        history = static_panel.loc[
            static_panel["player_id"].isin(missing_ids)
            & static_panel["season_start_year"].lt(target_year),
            [column for column in _BIOGRAPHY_COLUMNS if column in static_panel],
        ].copy()
        history = (
            history.sort_values(["player_id", "season_start_year"], kind="stable")
            .drop_duplicates("player_id", keep="last")
        )
        if not history.empty:
            history["age"] = pd.to_numeric(history["age"], errors="coerce") + (
                target_year - pd.to_numeric(history["season_start_year"], errors="coerce")
            )
            history["static_bio_source"] = "prior_static_biography"
            historical = history

    biography_frames = [target]
    if not historical.empty:
        biography_frames.append(historical)
    bios = pd.concat(biography_frames, ignore_index=True).drop_duplicates("player_id", keep="first")
    output = roster.merge(bios, on="player_id", how="left", validate="one_to_one")
    for column in ("age", "draft_year", "draft_number"):
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["listed_position"] = output["listed_position"].fillna("").astype(str)
    output["is_rookie"] = output["is_rookie"].astype("boolean").fillna(False).astype(bool)
    output["is_undrafted"] = output["is_undrafted"].astype("boolean").fillna(False).astype(bool)

    unresolved = output["static_bio_source"].isna()
    if unresolved.any():
        output.loc[unresolved, "age"] = output.loc[unresolved, "age"].fillna(22.0)
        output.loc[unresolved, "is_rookie"] = True
        output.loc[unresolved, "is_undrafted"] = output.loc[unresolved, "draft_number"].isna()
        output.loc[unresolved, "static_bio_source"] = "neutral_static_fallback"
    return output.loc[
        :,
        [
            "team_id",
            "team",
            "player_id",
            "player_name",
            "age",
            "draft_year",
            "draft_number",
            "is_undrafted",
            "listed_position",
            "is_rookie",
            "static_bio_source",
        ],
    ]


def build_forward_nail_roster_squash_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
) -> ForwardNailRosterSquashInputs:
    """Load a frozen target's opening inputs and the separate realized label."""

    opening_roster = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    static_bios = build_static_opening_roster_bios(
        season, opening_roster, panel=panel, curated_dir=curated_dir
    )
    return ForwardNailRosterSquashInputs(
        season=season,
        opening_roster=opening_roster,
        static_roster_bios=static_bios,
        preseason_nail=build_preseason_nail_forecast_ratings(
            season, opening_roster, panel=panel, run_dir=run_dir
        ),
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
    )


def assert_full_regular_season_minute_coverage(
    season: str,
    game_minutes: pd.DataFrame,
    *,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
) -> None:
    """Require the complete catalog before accepting a full-season score."""

    catalog = pd.read_parquet(catalog_path)
    required = {
        "season",
        "season_type",
        "game_id",
        "home_team_id",
        "away_team_id",
    }
    missing = sorted(required - set(catalog))
    if missing:
        raise ValueError("Game catalog lacks required columns: " + ", ".join(missing))
    expected = catalog.loc[
        catalog["season"].astype(str).eq(season) & catalog["season_type"].eq("regular"),
        ["game_id", "home_team_id", "away_team_id"],
    ].copy()
    if expected.empty:
        raise ValueError(f"Game catalog has no regular-season games for {season}")
    observed_game_ids = set(game_minutes["game_id"].astype(str))
    expected_game_ids = set(expected["game_id"].astype(str))
    missing_games = sorted(expected_game_ids - observed_game_ids)
    unexpected_games = sorted(observed_game_ids - expected_game_ids)
    expected_team_games = pd.concat(
        [
            expected.loc[:, ["game_id", "home_team_id"]].rename(
                columns={"home_team_id": "team_id"}
            ),
            expected.loc[:, ["game_id", "away_team_id"]].rename(
                columns={"away_team_id": "team_id"}
            ),
        ],
        ignore_index=True,
    ).groupby("team_id", as_index=True)["game_id"].nunique()
    observed_team_games = game_minutes.groupby("team_id", as_index=True)["game_id"].nunique()
    incomplete_teams = {
        int(team_id): {
            "expected": int(expected_count),
            "observed": int(observed_team_games.get(team_id, 0)),
        }
        for team_id, expected_count in expected_team_games.items()
        if int(observed_team_games.get(team_id, 0)) != int(expected_count)
    }
    if missing_games or unexpected_games or incomplete_teams:
        examples = ", ".join(missing_games[:5]) or "none"
        raise ValueError(
            f"{season} full-season minute target is incomplete: "
            f"missing_games={len(missing_games)} (examples: {examples}), "
            f"unexpected_games={len(unexpected_games)}, "
            f"incomplete_teams={len(incomplete_teams)}. Recover player-game coverage "
            "before running the all-season evaluator."
        )


def build_forward_conditional_raw_weights(
    inputs: ForwardNailRosterSquashInputs,
    *,
    availability_summary: pd.DataFrame,
    cold_start_config: ColdStartConditionalMinutesConfig = (
        PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG
    ),
) -> pd.DataFrame:
    """Materialize forward availability-times-conditional-minutes raw weights."""

    availability, _age, _metadata = predict_availability_roster(
        availability_summary,
        roster=inputs.static_roster_bios,
        target_season=inputs.season,
        config=PROMOTED_AVAILABILITY_CONFIG,
    )
    conditional, _conditional_age = predict_conditional_minutes_roster(
        availability_summary,
        roster=inputs.static_roster_bios,
        target_season=inputs.season,
        config=PROMOTED_CONDITIONAL_MINUTES_CONFIG,
        cold_start_config=cold_start_config,
    )
    combined = attach_expected_total_minutes(conditional, availability)
    combined = combined.drop(columns="player_name", errors="ignore")
    return inputs.static_roster_bios.loc[:, ["team_id", "team", "player_id", "player_name"]].merge(
        combined,
        on="player_id",
        how="left",
        validate="one_to_one",
    )


def apply_forward_nail_tilt(
    raw_predictions: pd.DataFrame,
    *,
    preseason_nail: pd.Series,
    nail_beta: float,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> pd.DataFrame:
    """Apply a NAIL tilt before the production top-15 roster squash.

    A target-season standard deviation of *preseason* NAIL values makes beta
    comparable across eras.  It is a valid forecast input because the entire
    preseason rating distribution exists before any target game is played.
    With beta zero, the result is exactly the production raw-weight and top-15
    selection contract.
    """

    required = {"team_id", "team", "player_id", "player_name", "raw_expected_total_minutes"}
    missing = sorted(required - set(raw_predictions))
    if missing:
        raise ValueError("Raw predictions lack required columns: " + ", ".join(missing))
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    if not np.isfinite(nail_beta):
        raise ValueError("NAIL beta must be finite")
    output = raw_predictions.copy()
    output["player_id"] = pd.to_numeric(output["player_id"], errors="raise").astype(int)
    ratings = pd.to_numeric(
        output["player_id"].map(preseason_nail.astype(float)), errors="coerce"
    ).fillna(0.0)
    rating_scale = float(np.std(ratings.to_numpy(dtype=float), ddof=0))
    if not np.isfinite(rating_scale) or rating_scale <= 1e-12:
        rating_scale = 1.0
    output["preseason_nail"] = ratings
    output["preseason_nail_z"] = ratings / rating_scale
    output["nail_rating_scale"] = rating_scale
    output["nail_tilt"] = np.exp(
        np.clip(float(nail_beta) * output["preseason_nail_z"].to_numpy(dtype=float), -30.0, 30.0)
    )
    output["base_raw_expected_total_minutes"] = pd.to_numeric(
        output["raw_expected_total_minutes"], errors="raise"
    )
    output["tilted_raw_expected_total_minutes"] = (
        output["base_raw_expected_total_minutes"] * output["nail_tilt"]
    )

    selected_ids: set[int] = set()
    for _, team in output.groupby("team_id", sort=False):
        selected_ids.update(
            team.sort_values(
                ["tilted_raw_expected_total_minutes", "player_name", "player_id"],
                ascending=[False, True, True],
                kind="stable",
            )
            .head(initial_rotation_size)["player_id"]
            .astype(int)
            .tolist()
        )
    output["is_baseline_rotation_candidate"] = output["player_id"].isin(selected_ids)
    output["raw_expected_total_minutes"] = output["tilted_raw_expected_total_minutes"].where(
        output["is_baseline_rotation_candidate"], 0.0
    )
    return output


def evaluate_forward_nail_roster_squash(
    inputs: ForwardNailRosterSquashInputs,
    *,
    availability_summary: pd.DataFrame,
    nail_beta: float,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
    team_games: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a preseason allocation against the full regular season by default.

    ``team_games`` is retained only for an explicitly requested diagnostic
    prefix. The candidate's selection and promotion target are all available
    regular-season games for each team.
    """

    if team_games is not None and team_games <= 0:
        raise ValueError("Team-games target must be positive")
    _validate_game_minutes(inputs.target_game_minutes, label="target")
    raw = build_forward_conditional_raw_weights(inputs, availability_summary=availability_summary)
    tilted = apply_forward_nail_tilt(
        raw,
        preseason_nail=inputs.preseason_nail,
        nail_beta=nail_beta,
        initial_rotation_size=initial_rotation_size,
    )
    opening_roster = inputs.opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]]
    normalized = normalize_opening_roster_minutes(tilted, opening_roster=opening_roster)
    predicted = opening_roster.merge(
        tilted.loc[
            :,
            [
                "player_id",
                "base_raw_expected_total_minutes",
                "tilted_raw_expected_total_minutes",
                "preseason_nail",
                "preseason_nail_z",
                "nail_tilt",
                "nail_rating_scale",
                "is_baseline_rotation_candidate",
            ],
        ],
        on="player_id",
        how="left",
        validate="one_to_one",
    ).merge(
        normalized.loc[:, ["player_id", "projected_minute_share", "projected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    targets = _season_allocation_targets(inputs.target_game_minutes, team_games=team_games)
    target_by_team = {target.team_id: target for target in targets}
    records: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    for team_id, team_predictions in predicted.groupby("team_id", sort=True):
        target = target_by_team.get(int(team_id))
        if target is None:
            raise ValueError(f"Target allocation missing team {team_id}")
        predicted_map = team_predictions.set_index("player_id")["projected_minute_share"].to_dict()
        actual_map = target.player_shares
        player_ids = sorted(set(predicted_map) | set(actual_map))
        actual_values = np.asarray(
            [float(actual_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        predicted_values = np.asarray(
            [float(predicted_map.get(player_id, 0.0)) for player_id in player_ids]
        )
        _assert_distribution(actual_values, label=f"actual for {target.team}")
        _assert_distribution(predicted_values, label=f"prediction for {target.team}")
        absolute_error = np.abs(actual_values - predicted_values)
        squared_error = np.square(actual_values - predicted_values)
        actual_top_ids = set(
            player_id
            for player_id, _share in sorted(
                actual_map.items(), key=lambda item: (-item[1], item[0])
            )[:initial_rotation_size]
        )
        selected_ids = set(
            team_predictions.loc[
                team_predictions["is_baseline_rotation_candidate"], "player_id"
            ].astype(int)
        )
        zero_probability_active = (actual_values > 0.0) & (predicted_values <= 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual_values > 0.0
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        metrics.append(
            {
                "season": inputs.season,
                "team_id": int(team_id),
                "team": target.team,
                "first_game_id": target.first_game_id,
                "last_game_id": target.last_game_id,
                "target_game_count": target.target_game_count,
                "nail_beta": float(nail_beta),
                "nail_rating_scale": float(team_predictions["nail_rating_scale"].iloc[0]),
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
                "outside_candidate_actual_share": float(
                    actual_values[
                        [player_id not in predicted_map for player_id in player_ids]
                    ].sum()
                ),
                "actual_top_rotation_overlap": float(
                    len(actual_top_ids & selected_ids) / max(len(actual_top_ids), 1)
                ),
                "selected_rotation_size": int(len(selected_ids)),
            }
        )
        player_rows = team_predictions.set_index("player_id")
        for player_id, actual_share, predicted_share in zip(
            player_ids, actual_values, predicted_values, strict=True
        ):
            source = player_rows.loc[player_id] if player_id in player_rows.index else None
            records.append(
                {
                    "season": inputs.season,
                    "team_id": int(team_id),
                    "team": target.team,
                    "player_id": int(player_id),
                    "player_name": (
                        str(source["player_name"]) if source is not None else str(player_id)
                    ),
                    "actual_minute_share": float(actual_share),
                    "predicted_minute_share": float(predicted_share),
                    "absolute_error": float(abs(actual_share - predicted_share)),
                    "base_raw_expected_total_minutes": (
                        float(source["base_raw_expected_total_minutes"])
                        if source is not None
                        else 0.0
                    ),
                    "tilted_raw_expected_total_minutes": (
                        float(source["tilted_raw_expected_total_minutes"])
                        if source is not None
                        else 0.0
                    ),
                    "preseason_nail": (
                        float(source["preseason_nail"]) if source is not None else 0.0
                    ),
                    "preseason_nail_z": (
                        float(source["preseason_nail_z"]) if source is not None else 0.0
                    ),
                    "nail_tilt": float(source["nail_tilt"]) if source is not None else 0.0,
                    "was_opening_candidate": source is not None,
                    "was_in_first_n_games": player_id in actual_map,
                    "was_selected_rotation": (
                        bool(source["is_baseline_rotation_candidate"])
                        if source is not None
                        else False
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metrics)


def select_forward_nail_beta(
    source_inputs: list[ForwardNailRosterSquashInputs],
    *,
    availability_summary: pd.DataFrame,
    nail_beta_grid: tuple[float, ...] = DEFAULT_NAIL_BETA_GRID,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
    team_games: int | None = None,
) -> tuple[float, pd.DataFrame]:
    """Select beta on pre-frozen seasons; beta zero is the exact control."""

    if not source_inputs:
        raise ValueError("NAIL beta selection requires at least one source season")
    if 0.0 not in nail_beta_grid:
        raise ValueError("NAIL beta grid must include the exact zero control")
    rows: list[dict[str, float | int]] = []
    for beta in sorted(set(float(value) for value in nail_beta_grid)):
        metrics = [
            evaluate_forward_nail_roster_squash(
                inputs,
                availability_summary=availability_summary,
                nail_beta=beta,
                initial_rotation_size=initial_rotation_size,
                team_games=team_games,
            )[1]
            for inputs in source_inputs
        ]
        combined = pd.concat(metrics, ignore_index=True)
        rows.append(
            {
                "nail_beta": beta,
                "source_season_count": len(source_inputs),
                "mean_allocation_total_variation": float(
                    combined["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(combined["brier_score"].mean()),
                "mean_player_share_mae": float(combined["player_share_mae"].mean()),
                "mean_actual_top_rotation_overlap": float(
                    combined["actual_top_rotation_overlap"].mean()
                ),
            }
        )
    grid = pd.DataFrame(rows)
    grid["absolute_nail_beta"] = grid["nail_beta"].abs()
    grid = grid.sort_values(
        [
            "mean_allocation_total_variation",
            "mean_brier_score",
            "absolute_nail_beta",
            "nail_beta",
        ],
        kind="stable",
    ).reset_index(drop=True)
    return float(grid.loc[0, "nail_beta"]), grid


def run_forward_nail_roster_squash(
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    run_dir: Path | str = DEFAULT_MODEL_RUN_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    nail_beta_grid: tuple[float, ...] = DEFAULT_NAIL_BETA_GRID,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
    team_games: int | None = None,
) -> ForwardNailRosterSquashRun:
    """Tune on pre-frozen seasons and freeze the NAIL tilt for three holdouts."""

    all_seasons = tuple(dict.fromkeys((*tuning_seasons, *frozen_seasons)))
    if not tuning_seasons or not frozen_seasons:
        raise ValueError("Both tuning and frozen seasons are required")
    if set(tuning_seasons) & set(frozen_seasons):
        raise ValueError("Tuning and frozen seasons must not overlap")
    panel = pd.read_parquet(panel_path)
    max_target_year = max(_season_year(season) for season in all_seasons)
    history_seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}" for year in range(2015, max_target_year)
    )
    print(
        f"[Forward NAIL squash] loading availability history through {history_seasons[-1]}",
        flush=True,
    )
    availability_summary = attach_cold_start_biographies(
        build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    )
    inputs_by_season: dict[str, ForwardNailRosterSquashInputs] = {}
    for season in all_seasons:
        print(f"[Forward NAIL squash] building frozen preseason inputs for {season}", flush=True)
        inputs_by_season[season] = build_forward_nail_roster_squash_inputs(
            season, panel=panel, curated_dir=curated_dir, run_dir=run_dir
        )
        if team_games is None:
            assert_full_regular_season_minute_coverage(
                season,
                inputs_by_season[season].target_game_minutes,
                catalog_path=catalog_path,
            )
    print(
        "[Forward NAIL squash] tuning beta on " + ", ".join(tuning_seasons), flush=True
    )
    selected_beta, tuning_grid = select_forward_nail_beta(
        [inputs_by_season[season] for season in tuning_seasons],
        availability_summary=availability_summary,
        nail_beta_grid=nail_beta_grid,
        initial_rotation_size=initial_rotation_size,
        team_games=team_games,
    )
    print(f"[Forward NAIL squash] selected beta={selected_beta:.3f}", flush=True)
    selected_predictions, selected_team_metrics = _evaluate_frozen_seasons(
        frozen_seasons,
        inputs_by_season=inputs_by_season,
        availability_summary=availability_summary,
        nail_beta=selected_beta,
        initial_rotation_size=initial_rotation_size,
        team_games=team_games,
        label="candidate",
    )
    control_predictions, control_team_metrics = _evaluate_frozen_seasons(
        frozen_seasons,
        inputs_by_season=inputs_by_season,
        availability_summary=availability_summary,
        nail_beta=0.0,
        initial_rotation_size=initial_rotation_size,
        team_games=team_games,
        label="beta-zero production control",
    )
    frozen_summary = _summarize_frozen_metrics(
        selected_team_metrics, frozen_seasons=frozen_seasons, nail_beta=selected_beta
    )
    control_summary = _summarize_frozen_metrics(
        control_team_metrics, frozen_seasons=frozen_seasons, nail_beta=0.0
    )
    metric_columns = (
        "allocation_total_variation",
        "brier_score",
        "player_share_mae",
        "player_share_mse",
        "actual_top_rotation_overlap",
    )
    frozen_comparison = selected_team_metrics.loc[:, ["season", "team_id", *metric_columns]].merge(
        control_team_metrics.loc[:, ["season", "team_id", *metric_columns]],
        on=["season", "team_id"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    for column in metric_columns:
        frozen_comparison[f"{column}_delta"] = (
            frozen_comparison[f"{column}_candidate"]
            - frozen_comparison[f"{column}_control"]
        )
    artifact_root = Path(artifacts_dir) / MODEL_NAME
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-nail-roster-squash-{timestamp}-{uuid4().hex[:7]}"
    output_dir = artifact_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    tuning_grid.to_parquet(output_dir / "tuning_grid.parquet", index=False)
    selected_predictions.to_parquet(output_dir / "frozen_predictions.parquet", index=False)
    selected_team_metrics.to_parquet(output_dir / "frozen_team_metrics.parquet", index=False)
    frozen_summary.to_parquet(output_dir / "frozen_summary.parquet", index=False)
    control_predictions.to_parquet(output_dir / "frozen_control_predictions.parquet", index=False)
    control_team_metrics.to_parquet(output_dir / "frozen_control_team_metrics.parquet", index=False)
    control_summary.to_parquet(output_dir / "frozen_control_summary.parquet", index=False)
    frozen_comparison.to_parquet(output_dir / "frozen_comparison.parquet", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "team_games": team_games,
                "evaluation_target": (
                    "all regular-season games for each team"
                    if team_games is None
                    else f"first {team_games} regular-season team games"
                ),
                "initial_rotation_size": initial_rotation_size,
                "nail_beta_grid": list(nail_beta_grid),
                "selected_nail_beta": selected_beta,
                "frozen_control_nail_beta": 0.0,
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "cold_start_conditional_minutes_config": asdict(
                    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG
                ),
                "contract": (
                    "At beta=0 this is the promoted forward availability times "
                    "conditional-minutes raw forecast, followed by its existing top-15 "
                    "opening-roster squash. Nonzero beta multiplies each raw forecast by "
                    "exp(beta times preseason NAIL divided by the target preseason NAIL "
                    "standard deviation), before top-15 selection and team normalization. "
                    "NAIL is reconstructed from the forward player prior and prior-season "
                    "frozen additive profile; target-season outcomes are used only for the "
                    "held-out full-regular-season minute-share label."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardNailRosterSquashRun(run_dir=output_dir, run_id=run_id)


def _season_year(season: str) -> int:
    return int(str(season)[:4])


def _season_allocation_targets(
    game_minutes: pd.DataFrame,
    *,
    team_games: int | None,
) -> list[_SeasonAllocationTarget]:
    """Return all-season targets, or an explicitly requested opening prefix."""

    if team_games is not None:
        return [
            _SeasonAllocationTarget(
                team_id=target.team_id,
                team=target.team,
                first_game_id=target.first_game_id,
                last_game_id=target.last_game_id,
                target_game_count=target.target_game_count,
                player_shares=target.player_shares,
            )
            for target in _first_n_team_game_allocations(game_minutes, team_games=team_games)
        ]
    ordered = game_minutes.sort_values(
        ["team_id", "game_time_utc", "game_id", "player_id"], kind="stable"
    )
    targets: list[_SeasonAllocationTarget] = []
    for team_id, rows in ordered.groupby("team_id", sort=True):
        games = rows.loc[:, ["game_id", "game_time_utc"]].drop_duplicates()
        player_minutes = rows.groupby("player_id", as_index=True)["minutes"].sum()
        total_minutes = float(player_minutes.sum())
        if total_minutes <= 0.0:
            raise ValueError(f"Regular season has no minutes for team {team_id}")
        targets.append(
            _SeasonAllocationTarget(
                team_id=int(team_id),
                team=str(rows["team_tricode"].iloc[0]),
                first_game_id=str(games["game_id"].iloc[0]),
                last_game_id=str(games["game_id"].iloc[-1]),
                target_game_count=int(len(games)),
                player_shares={
                    int(player_id): float(minutes / total_minutes)
                    for player_id, minutes in player_minutes.items()
                },
            )
        )
    return targets


def _evaluate_frozen_seasons(
    seasons: tuple[str, ...],
    *,
    inputs_by_season: dict[str, ForwardNailRosterSquashInputs],
    availability_summary: pd.DataFrame,
    nail_beta: float,
    initial_rotation_size: int,
    team_games: int | None,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for season in seasons:
        print(f"[Forward NAIL squash] freezing {season} ({label})", flush=True)
        predictions, metrics = evaluate_forward_nail_roster_squash(
            inputs_by_season[season],
            availability_summary=availability_summary,
            nail_beta=nail_beta,
            initial_rotation_size=initial_rotation_size,
            team_games=team_games,
        )
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.concat(metric_frames, ignore_index=True),
    )


def _summarize_frozen_metrics(
    metrics: pd.DataFrame,
    *,
    frozen_seasons: tuple[str, ...],
    nail_beta: float,
) -> pd.DataFrame:
    summaries = [
        summarize_l20_metrics(metrics.loc[metrics["season"].eq(season)], season=season).assign(
            nail_beta=nail_beta
        )
        for season in frozen_seasons
    ]
    finite_cross_entropy = metrics.loc[np.isfinite(metrics["cross_entropy"]), "cross_entropy"]
    summaries.append(
        pd.DataFrame(
            [
                {
                    "season": "pooled_frozen",
                    "evaluated_teams": int(len(metrics)),
                    "mean_allocation_total_variation": float(
                        metrics["allocation_total_variation"].mean()
                    ),
                    "mean_brier_score": float(metrics["brier_score"].mean()),
                    "player_share_mae": float(metrics["player_share_mae"].mean()),
                    "player_share_rmse": float(np.sqrt(metrics["player_share_mse"].mean())),
                    "strict_cross_entropy": float(metrics["cross_entropy"].mean()),
                    "share_teams_infinite_cross_entropy": float(
                        metrics["has_zero_probability_active_player"].mean()
                    ),
                    "mean_cross_entropy_when_finite": float(finite_cross_entropy.mean()),
                    "mean_outside_candidate_actual_share": float(
                        metrics["outside_candidate_actual_share"].mean()
                    ),
                    "nail_beta": nail_beta,
                }
            ]
        )
    )
    return pd.concat(summaries, ignore_index=True)


def main() -> None:
    """Run the forward NAIL roster-squash frozen evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a forward NAIL roster squash")
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument(
        "--team-games",
        type=int,
        default=None,
        help="Use only this many opening games for a diagnostic; default is the full season.",
    )
    parser.add_argument("--initial-rotation-size", type=int, default=DEFAULT_INITIAL_ROTATION_SIZE)
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_MODEL_RUN_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()
    run = run_forward_nail_roster_squash(
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        team_games=args.team_games,
        initial_rotation_size=args.initial_rotation_size,
        panel_path=args.panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        run_dir=args.run_dir,
        catalog_path=args.catalog_path,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
