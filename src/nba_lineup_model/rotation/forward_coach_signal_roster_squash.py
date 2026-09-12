"""Forward-safe coach-signal adjustment to the promoted preseason roster squash.

The production forecast supplies independent expected totals from Forward
Availability and Forward Conditional Minutes. This candidate tests two
preseason hierarchy signals before the existing active-15 roster constraint:
the prior season's shrunken start rate and recent draft investment. The
all-zero coefficient configuration exactly recovers the production squash.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from nba_lineup_model.rotation.forward_availability import (
    DEFAULT_CURATED_DIR,
    PROMOTED_AVAILABILITY_CONFIG,
    build_availability_season_summary,
)
from nba_lineup_model.rotation.forward_conditional_minutes import (
    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG,
    PROMOTED_CONDITIONAL_MINUTES_CONFIG,
    attach_cold_start_biographies,
    build_forward_conditional_roster_profile,
    normalize_opening_roster_minutes,
)
from nba_lineup_model.rotation.forward_nail_roster_squash import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_INITIAL_ROTATION_SIZE,
    _season_allocation_targets,
    assert_full_regular_season_minute_coverage,
    build_forward_conditional_raw_weights,
    build_static_opening_roster_bios,
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
    _validate_game_minutes,
)
from nba_lineup_model.rotation.l20_preseason_nail_forecast_minute_share import (
    DEFAULT_PANEL_PATH,
)

MODEL_NAME = "forward_coach_signal_roster_squash"
MODEL_VERSION = "v0.1"
DEFAULT_TUNING_SEASONS = ("2020-21", "2021-22", "2022-23")
DEFAULT_FROZEN_SEASONS = ("2023-24", "2024-25", "2025-26")
DEFAULT_START_WEIGHT_GRID = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
DEFAULT_DRAFT_WEIGHT_GRID = (0.0, 0.05, 0.10, 0.20, 0.40, 0.80)
DEFAULT_START_SHRINKAGE_GRID = (5.0, 15.0, 30.0)
DEFAULT_DRAFT_HALF_LIFE_GRID = (1.0, 2.0, 3.0, 4.0)


@dataclass(frozen=True)
class CoachSignalConfig:
    """Positive pre-squash weights and their source-season tuning constants."""

    start_weight: float
    draft_weight: float
    start_shrinkage_games: float
    draft_half_life_years: float


@dataclass(frozen=True)
class ForwardCoachSignalInputs:
    """Frozen preseason roster state and the distinct realized minute label."""

    season: str
    opening_roster: pd.DataFrame
    static_roster_bios: pd.DataFrame
    prior_player_season: pd.DataFrame
    target_game_minutes: pd.DataFrame


@dataclass(frozen=True)
class ForwardCoachSignalRun:
    """Immutable output location for one coach-signal frozen replay."""

    run_dir: Path
    run_id: str


def build_forward_coach_signal_inputs(
    season: str,
    *,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> ForwardCoachSignalInputs:
    """Create a target-season input contract using only preseason information."""

    opening_roster = read_preseason_roster_candidates(season, curated_dir=curated_dir)
    static_bios = build_static_opening_roster_bios(
        season, opening_roster, panel=panel, curated_dir=curated_dir
    )
    prior_year = _season_year(season) - 1
    prior_columns = ["player_id", "games", "games_started"]
    missing = sorted(set(prior_columns) - set(panel))
    if missing:
        raise ValueError(
            "Player-season panel lacks prior coach-signal fields: " + ", ".join(missing)
        )
    prior = panel.loc[
        pd.to_numeric(panel["season_start_year"], errors="raise").eq(prior_year),
        prior_columns,
    ].copy()
    prior["player_id"] = pd.to_numeric(prior["player_id"], errors="raise").astype(int)
    if prior.duplicated("player_id").any():
        raise ValueError(f"Prior panel has duplicate player rows for {season}")
    return ForwardCoachSignalInputs(
        season=season,
        opening_roster=opening_roster,
        static_roster_bios=static_bios,
        prior_player_season=prior,
        target_game_minutes=read_regular_game_minutes(season, curated_dir=curated_dir),
    )


def build_live_forward_coach_signal_inputs(
    *,
    season: str,
    roster_path: Path | str,
    panel: pd.DataFrame,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
) -> ForwardCoachSignalInputs:
    """Build current-season candidate inputs from the served roster contract."""

    roster = pd.read_parquet(roster_path)
    required = {
        "team_id",
        "team_abbreviation",
        "player_id",
        "player_name",
        "age",
        "listed_position",
    }
    missing = sorted(required - set(roster))
    if missing:
        raise ValueError("Roster lacks required current forecast columns: " + ", ".join(missing))
    opening_roster = roster.rename(columns={"team_abbreviation": "team"}).loc[
        :, ["team_id", "team", "player_id", "player_name"]
    ].copy()
    opening_roster["team_id"] = pd.to_numeric(
        opening_roster["team_id"], errors="raise"
    ).astype(int)
    opening_roster["player_id"] = pd.to_numeric(
        opening_roster["player_id"], errors="raise"
    ).astype(int)
    profile = build_forward_conditional_roster_profile(
        roster, target_season=season, curated_dir=curated_dir
    )
    draft_path = Path(curated_dir) / "draft_history" / season / "part-00000.parquet"
    if draft_path.exists():
        draft_year = pd.read_parquet(draft_path, columns=["player_id", "draft_year"])
        draft_year["player_id"] = pd.to_numeric(
            draft_year["player_id"], errors="raise"
        ).astype(int)
        draft_year = draft_year.drop_duplicates("player_id", keep="last")
    else:
        draft_year = pd.DataFrame(columns=["player_id", "draft_year"])
    static_bios = (
        opening_roster.merge(
            profile,
            on=["player_id", "player_name"],
            how="left",
            validate="one_to_one",
        )
        .merge(draft_year, on="player_id", how="left", validate="one_to_one")
    )
    prior_year = _season_year(season) - 1
    prior = panel.loc[
        pd.to_numeric(panel["season_start_year"], errors="raise").eq(prior_year),
        ["player_id", "games", "games_started"],
    ].copy()
    prior["player_id"] = pd.to_numeric(prior["player_id"], errors="raise").astype(int)
    if prior.duplicated("player_id").any():
        raise ValueError(f"Prior panel has duplicate player rows for {season}")
    return ForwardCoachSignalInputs(
        season=season,
        opening_roster=opening_roster,
        static_roster_bios=static_bios,
        prior_player_season=prior,
        target_game_minutes=pd.DataFrame(),
    )


def build_coach_signal_features(
    inputs: ForwardCoachSignalInputs,
    *,
    config: CoachSignalConfig,
) -> pd.DataFrame:
    """Build team-relative start and draft hierarchy features before target play.

    Start rate is the prior season's GS / GP posterior under a league start-rate
    prior. Recent draft investment is normalized draft capital exponentially
    decayed by years since draft. Both features are z-scored only within their
    prospective roster because the subsequent squash is a within-team choice.
    """

    if config.start_shrinkage_games <= 0.0:
        raise ValueError("Start shrinkage must be positive")
    if config.draft_half_life_years <= 0.0:
        raise ValueError("Draft half-life must be positive")
    roster = inputs.static_roster_bios.loc[
        :,
        [
            "team_id",
            "team",
            "player_id",
            "player_name",
            "draft_year",
            "draft_number",
        ],
    ].copy()
    roster["player_id"] = pd.to_numeric(roster["player_id"], errors="raise").astype(int)
    prior = inputs.prior_player_season.copy()
    prior["games"] = pd.to_numeric(prior["games"], errors="coerce").fillna(0.0)
    prior["games_started"] = pd.to_numeric(prior["games_started"], errors="coerce").fillna(0.0)
    total_games = float(prior["games"].sum())
    league_start_rate = (
        float(prior["games_started"].sum() / total_games) if total_games > 0.0 else 0.35
    )
    output = roster.merge(prior, on="player_id", how="left", validate="one_to_one")
    output["prior_games"] = output["games"].fillna(0.0).clip(lower=0.0)
    output["prior_games_started"] = output["games_started"].fillna(0.0).clip(lower=0.0)
    output["shrunken_start_rate"] = (
        output["prior_games_started"]
        + config.start_shrinkage_games * league_start_rate
    ) / (output["prior_games"] + config.start_shrinkage_games)
    draft_number = pd.to_numeric(output["draft_number"], errors="coerce")
    draft_year = pd.to_numeric(output["draft_year"], errors="coerce")
    target_year = _season_year(inputs.season)
    years_since_draft = (target_year - draft_year).clip(lower=0.0)
    draft_capital = np.where(
        draft_number.notna(), np.clip((61.0 - draft_number) / 60.0, 0.0, 1.0), 0.0
    )
    output["years_since_draft"] = years_since_draft
    output["recent_draft_investment"] = draft_capital * np.exp(
        -years_since_draft / config.draft_half_life_years
    )
    output["start_rate_z"] = _within_team_zscore(output, "shrunken_start_rate")
    output["draft_investment_z"] = _within_team_zscore(output, "recent_draft_investment")
    output["league_start_rate"] = league_start_rate
    return output.loc[
        :,
        [
            "team_id",
            "team",
            "player_id",
            "player_name",
            "prior_games",
            "prior_games_started",
            "league_start_rate",
            "shrunken_start_rate",
            "start_rate_z",
            "years_since_draft",
            "recent_draft_investment",
            "draft_investment_z",
        ],
    ]


def _within_team_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    """Standardize a signal only among the players competing for one rotation."""

    result = pd.Series(0.0, index=frame.index, dtype=float)
    for _, indexes in frame.groupby("team_id", sort=False).groups.items():
        values = pd.to_numeric(frame.loc[indexes, column], errors="coerce").fillna(0.0)
        scale = float(values.std(ddof=0))
        if np.isfinite(scale) and scale > 1e-12:
            result.loc[indexes] = (values - float(values.mean())) / scale
    return result


def apply_forward_coach_signals(
    raw_predictions: pd.DataFrame,
    *,
    coach_signals: pd.DataFrame,
    config: CoachSignalConfig,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> pd.DataFrame:
    """Reweight raw forecast totals and then apply the unchanged active-15 gate."""

    required = {"team_id", "team", "player_id", "player_name", "raw_expected_total_minutes"}
    missing = sorted(required - set(raw_predictions))
    if missing:
        raise ValueError("Raw predictions lack required columns: " + ", ".join(missing))
    if initial_rotation_size <= 0:
        raise ValueError("Initial rotation size must be positive")
    if config.start_weight < 0.0 or config.draft_weight < 0.0:
        raise ValueError("Coach-signal weights must be non-negative")
    output = raw_predictions.merge(
        coach_signals,
        on=["team_id", "team", "player_id", "player_name"],
        how="left",
        validate="one_to_one",
    )
    if output["start_rate_z"].isna().any() or output["draft_investment_z"].isna().any():
        raise ValueError("Raw predictions lack matched coach signals")
    output["base_raw_expected_total_minutes"] = pd.to_numeric(
        output["raw_expected_total_minutes"], errors="raise"
    )
    output["coach_signal_score"] = (
        config.start_weight * output["start_rate_z"]
        + config.draft_weight * output["draft_investment_z"]
    )
    output["coach_signal_tilt"] = np.exp(
        np.clip(output["coach_signal_score"].to_numpy(dtype=float), -30.0, 30.0)
    )
    output["tilted_raw_expected_total_minutes"] = (
        output["base_raw_expected_total_minutes"] * output["coach_signal_tilt"]
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
    output["is_rotation_candidate"] = output["player_id"].isin(selected_ids)
    output["raw_expected_total_minutes"] = output[
        "tilted_raw_expected_total_minutes"
    ].where(output["is_rotation_candidate"], 0.0)
    return output


def evaluate_forward_coach_signal_roster_squash(
    inputs: ForwardCoachSignalInputs,
    *,
    availability_summary: pd.DataFrame,
    config: CoachSignalConfig,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
    raw_predictions: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score one frozen preseason allocation against every regular-season game."""

    _validate_game_minutes(inputs.target_game_minutes, label="target")
    raw = (
        raw_predictions.copy()
        if raw_predictions is not None
        else build_forward_conditional_raw_weights(
            inputs, availability_summary=availability_summary
        )
    )
    signals = build_coach_signal_features(inputs, config=config)
    adjusted = apply_forward_coach_signals(
        raw,
        coach_signals=signals,
        config=config,
        initial_rotation_size=initial_rotation_size,
    )
    opening_roster = inputs.opening_roster.loc[:, ["team_id", "team", "player_id", "player_name"]]
    normalized = normalize_opening_roster_minutes(adjusted, opening_roster=opening_roster)
    predicted = opening_roster.merge(
        adjusted,
        on=["team_id", "team", "player_id", "player_name"],
        how="left",
        validate="one_to_one",
    ).merge(
        normalized.loc[:, ["player_id", "projected_minute_share", "projected_total_minutes"]],
        on="player_id",
        how="left",
        validate="one_to_one",
    )
    targets = _season_allocation_targets(inputs.target_game_minutes, team_games=None)
    target_by_team = {target.team_id: target for target in targets}
    records: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
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
        actual_top_ids = {
            player_id
            for player_id, _share in sorted(
                actual_map.items(), key=lambda item: (-item[1], item[0])
            )[:initial_rotation_size]
        }
        selected_ids = set(
            team_predictions.loc[team_predictions["is_rotation_candidate"], "player_id"]
            .astype(int)
            .tolist()
        )
        zero_probability_active = (actual_values > 0.0) & (predicted_values <= 0.0)
        cross_entropy = float("inf")
        if not zero_probability_active.any():
            positive_actual = actual_values > 0.0
            cross_entropy = float(
                -(actual_values[positive_actual] * np.log(predicted_values[positive_actual])).sum()
            )
        metric_rows.append(
            {
                "season": inputs.season,
                "team_id": int(team_id),
                "team": target.team,
                "target_game_count": target.target_game_count,
                **asdict(config),
                "allocation_total_variation": float(0.5 * absolute_error.sum()),
                "brier_score": float(squared_error.sum()),
                "player_share_mae": float(absolute_error.mean()),
                "player_share_mse": float(squared_error.mean()),
                "cross_entropy": cross_entropy,
                "has_zero_probability_active_player": bool(zero_probability_active.any()),
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
                    "coach_signal_score": (
                        float(source["coach_signal_score"]) if source is not None else 0.0
                    ),
                    "coach_signal_tilt": (
                        float(source["coach_signal_tilt"]) if source is not None else 0.0
                    ),
                    "shrunken_start_rate": (
                        float(source["shrunken_start_rate"]) if source is not None else 0.0
                    ),
                    "recent_draft_investment": (
                        float(source["recent_draft_investment"]) if source is not None else 0.0
                    ),
                    "was_selected_rotation": (
                        bool(source["is_rotation_candidate"]) if source is not None else False
                    ),
                }
            )
    return pd.DataFrame(records), pd.DataFrame(metric_rows)


def select_coach_signal_config(
    source_inputs: list[ForwardCoachSignalInputs],
    *,
    availability_summary: pd.DataFrame,
    raw_predictions_by_season: dict[str, pd.DataFrame],
    start_weight_grid: tuple[float, ...] = DEFAULT_START_WEIGHT_GRID,
    draft_weight_grid: tuple[float, ...] = DEFAULT_DRAFT_WEIGHT_GRID,
    start_shrinkage_grid: tuple[float, ...] = DEFAULT_START_SHRINKAGE_GRID,
    draft_half_life_grid: tuple[float, ...] = DEFAULT_DRAFT_HALF_LIFE_GRID,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> tuple[CoachSignalConfig, pd.DataFrame]:
    """Select the smallest forward-safe coach-signal configuration by source TV."""

    rows: list[dict[str, float | int]] = []
    for values in product(
        sorted(set(start_weight_grid)),
        sorted(set(draft_weight_grid)),
        sorted(set(start_shrinkage_grid)),
        sorted(set(draft_half_life_grid)),
    ):
        config = CoachSignalConfig(*map(float, values))
        metrics = [
            evaluate_forward_coach_signal_roster_squash(
                inputs,
                availability_summary=availability_summary,
                config=config,
                initial_rotation_size=initial_rotation_size,
                raw_predictions=raw_predictions_by_season[inputs.season],
            )[1]
            for inputs in source_inputs
        ]
        combined = pd.concat(metrics, ignore_index=True)
        rows.append(
            {
                **asdict(config),
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
    grid["weight_l1"] = grid["start_weight"] + grid["draft_weight"]
    grid = grid.sort_values(
        [
            "mean_allocation_total_variation",
            "mean_brier_score",
            "mean_player_share_mae",
            "weight_l1",
            "start_shrinkage_games",
            "draft_half_life_years",
        ],
        kind="stable",
    ).reset_index(drop=True)
    winner = grid.loc[0]
    return (
        CoachSignalConfig(
            start_weight=float(winner["start_weight"]),
            draft_weight=float(winner["draft_weight"]),
            start_shrinkage_games=float(winner["start_shrinkage_games"]),
            draft_half_life_years=float(winner["draft_half_life_years"]),
        ),
        grid,
    )


def run_forward_coach_signal_roster_squash(
    *,
    tuning_seasons: tuple[str, ...] = DEFAULT_TUNING_SEASONS,
    frozen_seasons: tuple[str, ...] = DEFAULT_FROZEN_SEASONS,
    panel_path: Path | str = DEFAULT_PANEL_PATH,
    curated_dir: Path | str = DEFAULT_CURATED_DIR,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    catalog_path: Path | str = DEFAULT_CATALOG_PATH,
    initial_rotation_size: int = DEFAULT_INITIAL_ROTATION_SIZE,
) -> ForwardCoachSignalRun:
    """Tune coach signals before freezing one nested candidate on three seasons."""

    if not tuning_seasons or not frozen_seasons:
        raise ValueError("Both tuning and frozen seasons are required")
    if set(tuning_seasons) & set(frozen_seasons):
        raise ValueError("Tuning and frozen seasons must not overlap")
    all_seasons = tuple(dict.fromkeys((*tuning_seasons, *frozen_seasons)))
    panel = pd.read_parquet(panel_path)
    max_target_year = max(_season_year(season) for season in all_seasons)
    history_seasons = tuple(
        f"{year}-{str(year + 1)[-2:]}" for year in range(2015, max_target_year)
    )
    print(
        f"[Coach signals] loading availability history through {history_seasons[-1]}",
        flush=True,
    )
    availability_summary = attach_cold_start_biographies(
        build_availability_season_summary(history_seasons, curated_dir=curated_dir)
    )
    inputs_by_season: dict[str, ForwardCoachSignalInputs] = {}
    raw_predictions_by_season: dict[str, pd.DataFrame] = {}
    for season in all_seasons:
        print(f"[Coach signals] building frozen preseason inputs for {season}", flush=True)
        inputs = build_forward_coach_signal_inputs(season, panel=panel, curated_dir=curated_dir)
        assert_full_regular_season_minute_coverage(
            season, inputs.target_game_minutes, catalog_path=catalog_path
        )
        inputs_by_season[season] = inputs
        raw_predictions_by_season[season] = build_forward_conditional_raw_weights(
            inputs, availability_summary=availability_summary
        )
    print("[Coach signals] tuning starts and recent draft investment", flush=True)
    selected_config, tuning_grid = select_coach_signal_config(
        [inputs_by_season[season] for season in tuning_seasons],
        availability_summary=availability_summary,
        raw_predictions_by_season=raw_predictions_by_season,
        initial_rotation_size=initial_rotation_size,
    )
    print(f"[Coach signals] selected {asdict(selected_config)}", flush=True)
    control_config = CoachSignalConfig(
        start_weight=0.0,
        draft_weight=0.0,
        start_shrinkage_games=selected_config.start_shrinkage_games,
        draft_half_life_years=selected_config.draft_half_life_years,
    )
    candidate_results = [
        evaluate_forward_coach_signal_roster_squash(
            inputs_by_season[season],
            availability_summary=availability_summary,
            config=selected_config,
            initial_rotation_size=initial_rotation_size,
            raw_predictions=raw_predictions_by_season[season],
        )
        for season in frozen_seasons
    ]
    control_results = [
        evaluate_forward_coach_signal_roster_squash(
            inputs_by_season[season],
            availability_summary=availability_summary,
            config=control_config,
            initial_rotation_size=initial_rotation_size,
            raw_predictions=raw_predictions_by_season[season],
        )
        for season in frozen_seasons
    ]
    candidate_predictions = pd.concat(
        [result[0] for result in candidate_results], ignore_index=True
    )
    candidate_metrics = pd.concat([result[1] for result in candidate_results], ignore_index=True)
    control_predictions = pd.concat([result[0] for result in control_results], ignore_index=True)
    control_metrics = pd.concat([result[1] for result in control_results], ignore_index=True)
    metric_columns = (
        "allocation_total_variation",
        "brier_score",
        "player_share_mae",
        "player_share_mse",
        "actual_top_rotation_overlap",
    )
    comparison = candidate_metrics.loc[:, ["season", "team_id", *metric_columns]].merge(
        control_metrics.loc[:, ["season", "team_id", *metric_columns]],
        on=["season", "team_id"],
        suffixes=("_candidate", "_control"),
        validate="one_to_one",
    )
    for column in metric_columns:
        comparison[f"{column}_delta"] = (
            comparison[f"{column}_candidate"] - comparison[f"{column}_control"]
        )
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"forward-coach-signal-roster-squash-{timestamp}-{uuid4().hex[:7]}"
    output_dir = Path(artifacts_dir) / MODEL_NAME / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    tuning_grid.to_parquet(output_dir / "tuning_grid.parquet", index=False)
    candidate_predictions.to_parquet(output_dir / "frozen_predictions.parquet", index=False)
    candidate_metrics.to_parquet(output_dir / "frozen_team_metrics.parquet", index=False)
    control_predictions.to_parquet(output_dir / "frozen_control_predictions.parquet", index=False)
    control_metrics.to_parquet(output_dir / "frozen_control_team_metrics.parquet", index=False)
    comparison.to_parquet(output_dir / "frozen_comparison.parquet", index=False)
    _summarize_coach_metrics(
        candidate_metrics,
        frozen_seasons=frozen_seasons,
        config=selected_config,
    ).to_parquet(output_dir / "frozen_summary.parquet", index=False)
    _summarize_coach_metrics(
        control_metrics,
        frozen_seasons=frozen_seasons,
        config=control_config,
    ).to_parquet(output_dir / "frozen_control_summary.parquet", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "version": MODEL_VERSION,
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "tuning_seasons": list(tuning_seasons),
                "frozen_seasons": list(frozen_seasons),
                "evaluation_target": "all regular-season games for each team",
                "initial_rotation_size": initial_rotation_size,
                "selected_config": asdict(selected_config),
                "control_config": asdict(control_config),
                "availability_config": asdict(PROMOTED_AVAILABILITY_CONFIG),
                "conditional_minutes_config": asdict(PROMOTED_CONDITIONAL_MINUTES_CONFIG),
                "cold_start_conditional_minutes_config": asdict(
                    PROMOTED_COLD_START_CONDITIONAL_MINUTES_CONFIG
                ),
                "contract": (
                    "Production raw expected-minute weights are multiplied by exp(start_weight "
                    "times within-roster shrunken prior GS/GP z-score plus draft_weight "
                    "times within-roster recent draft-investment z-score), then pass through "
                    "the unchanged top-15 and team-minute normalization. Prior GS/GP comes "
                    "only from the immediately preceding completed season. Draft investment "
                    "uses preexisting draft capital decayed by years since draft. Target-season "
                    "outcomes appear only in the full regular-season minute label."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return ForwardCoachSignalRun(run_dir=output_dir, run_id=run_id)


def _season_year(season: str) -> int:
    return int(season[:4])


def _summarize_coach_metrics(
    metrics: pd.DataFrame,
    *,
    frozen_seasons: tuple[str, ...],
    config: CoachSignalConfig,
) -> pd.DataFrame:
    """Summarize season-level and pooled metrics with the selected configuration."""

    rows: list[dict[str, object]] = []
    for season in (*frozen_seasons, "pooled_frozen"):
        source = metrics if season == "pooled_frozen" else metrics.loc[metrics["season"].eq(season)]
        rows.append(
            {
                "season": season,
                "evaluated_teams": int(len(source)),
                "mean_allocation_total_variation": float(
                    source["allocation_total_variation"].mean()
                ),
                "mean_brier_score": float(source["brier_score"].mean()),
                "player_share_mae": float(source["player_share_mae"].mean()),
                "player_share_rmse": float(np.sqrt(source["player_share_mse"].mean())),
                "mean_actual_top_rotation_overlap": float(
                    source["actual_top_rotation_overlap"].mean()
                ),
                **asdict(config),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the full-season frozen coach-signals evaluation."""

    parser = argparse.ArgumentParser(
        description="Evaluate coach signals in preseason rotation squash"
    )
    parser.add_argument("--tuning-seasons", nargs="+", default=list(DEFAULT_TUNING_SEASONS))
    parser.add_argument("--frozen-seasons", nargs="+", default=list(DEFAULT_FROZEN_SEASONS))
    parser.add_argument("--panel-path", type=Path, default=DEFAULT_PANEL_PATH)
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--catalog-path", type=Path, default=DEFAULT_CATALOG_PATH)
    args = parser.parse_args()
    run = run_forward_coach_signal_roster_squash(
        tuning_seasons=tuple(args.tuning_seasons),
        frozen_seasons=tuple(args.frozen_seasons),
        panel_path=args.panel_path,
        curated_dir=args.curated_dir,
        artifacts_dir=args.artifacts_dir,
        catalog_path=args.catalog_path,
    )
    print(run.run_dir)


if __name__ == "__main__":
    main()
